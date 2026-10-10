#!/usr/bin/env python3
"""
Decrypt Authy GDPR export tokens and convert to standard authenticator formats.

Encryption scheme:
  Key  = PBKDF2-HMAC-SHA1(password, salt_utf8, iterations=100000, dklen=32)
  Data = AES-256-CBC(key, iv_hex, base64(encrypted_seed))
"""
import argparse
import base64
import binascii
import csv
import getpass
import hashlib
import io
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

from Crypto.Cipher import AES


KDF_ITERATIONS = 100_000
REQUIRED_COLUMNS = {"name", "encrypted_seed", "salt", "iv"}


def decrypt_seed(encrypted_seed_b64: str, salt: str, iv_hex: str, password: str) -> str:
    """Return base32-encoded TOTP secret (uppercase)."""
    key = hashlib.pbkdf2_hmac(
        "sha1",
        password.encode("utf-8"),
        salt.encode("utf-8"),  # salt is raw UTF-8, not base64
        KDF_ITERATIONS,
        dklen=32,
    )
    iv = bytes.fromhex(iv_hex)
    ciphertext = base64.b64decode(encrypted_seed_b64)
    plain = AES.new(key, AES.MODE_CBC, iv).decrypt(ciphertext)

    pad = plain[-1]
    if pad < 1 or pad > 16 or not all(b == pad for b in plain[-pad:]):
        raise ValueError("Invalid padding — wrong password?")

    return plain[:-pad].decode("utf-8").strip().upper()


def _unwrap_quotes(text: str) -> str:
    return text[1:-1] if len(text) >= 2 and text.startswith('"') and text.endswith('"') else text


def normalize_authy_csv(content: str) -> str:
    """Strip the stray quoting Authy adds to its CSV export.

    Older exports wrap the entire file in a single pair of quotes; newer ones
    wrap each line in its own pair. Either way Python's csv parser would
    treat the quoted span as one field. Detect the format from the header
    before removing wrappers, preserving ordinary CSV field quotes.
    """
    content = content.strip()
    if not content:
        return content

    header = content.splitlines()[0]

    def valid_header(value: str) -> bool:
        # strict=True makes an unclosed quote raise, so a header that has
        # been only partially unwrapped (e.g. the whole-file format with
        # quoted field names) never validates in the wrong branch below.
        try:
            fields = next(csv.reader([value], strict=True))
        except (csv.Error, StopIteration):
            return False
        return REQUIRED_COLUMNS.issubset(fields)

    if valid_header(header):
        return content
    if header.startswith('"') and header.endswith('"') and valid_header(header[1:-1]):
        return "\n".join(_unwrap_quotes(line) for line in content.splitlines())
    if content.startswith('"') and content.endswith('"') and valid_header(header[1:]):
        return _unwrap_quotes(content)
    return content


def parse_row(header: list[str], fields: list[str]) -> dict[str, str]:
    """Map a parsed CSV row onto the header, anchored on the right.

    Authy writes token names without CSV quoting, so a name containing a
    comma arrives as extra leading fields. When the header ends with the
    encrypted_seed, salt and iv columns, the surplus fields are joined back
    into the name.
    """
    if len(fields) < len(header):
        raise ValueError(
            f"malformed row: expected {len(header)} fields, got {len(fields)}"
        )
    surplus = len(fields) - len(header)
    if surplus and "name" in header and header[-3:] == ["encrypted_seed", "salt", "iv"]:
        i = header.index("name")
        fields = fields[:i] + [",".join(fields[i : i + surplus + 1])] + fields[i + surplus + 1 :]
    return dict(zip(header, fields))


def split_name(name_field: str) -> tuple[str, str]:
    """Split 'Service:account' into (issuer, account)."""
    issuer, _, account = name_field.partition(":")
    return issuer.strip(), account.strip()


def build_otpauth(issuer: str, account: str, secret: str) -> str:
    issuer_enc = quote(issuer, safe="")
    label = f"{issuer_enc}:{quote(account, safe='')}" if account else issuer_enc
    return f"otpauth://totp/{label}?secret={secret}&issuer={issuer_enc}&algorithm=SHA1&digits=6&period=30"


def write_secret_file(path: Path, content: str) -> None:
    """Write file with owner-only (0600) permissions to avoid leaking secrets to other local users."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # fchmod before writing closes the race window when the file pre-existed with looser perms:
    # os.open's mode arg is ignored on existing files, so the first write would otherwise
    # land while the file is still readable by others.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)


def build_aegis(tokens: list[dict]) -> dict:
    # Aegis vault db v3. Every entry key is emitted even when empty: some
    # importers (Proton Authenticator) deserialize strictly and reject the
    # whole file if e.g. "groups" is absent.
    return {
        "version": 1,
        "header": {"slots": None, "params": None},
        "db": {
            "version": 3,
            "entries": [
                {
                    "type": "totp",
                    "uuid": str(uuid.uuid4()),
                    "name": t["account"] or t["issuer"],
                    "issuer": t["issuer"],
                    "note": "",
                    "favorite": False,
                    "icon": None,
                    "info": {
                        "secret": t["secret"],
                        "algo": "SHA1",
                        "digits": 6,
                        "period": 30,
                    },
                    "groups": [],
                }
                for t in tokens
            ],
            "groups": [],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decrypt Authy GDPR export tokens into standard authenticator formats."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the Authy GDPR export CSV file"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same directory as input)",
    )
    parser.add_argument(
        "-p",
        "--password",
        default=None,
        help=(
            "Backup password. UNSAFE: visible to other users via `ps` "
            "and stored in shell history. Omit to be prompted."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["aegis", "uris", "both"],
        default="both",
        help="Output format: aegis JSON, otpauth URI list, or both (default: both)",
    )
    parser.add_argument(
        "--aegis-file",
        default="tokens_aegis.json",
        metavar="NAME",
        help="Filename for the Aegis JSON export (default: tokens_aegis.json)",
    )
    parser.add_argument(
        "--uris-file",
        default="otpauth_uris.txt",
        metavar="NAME",
        help="Filename for the otpauth URI list (default: otpauth_uris.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.input.parent
    if not output_dir.is_dir():
        print(f"Error: output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    password = args.password or getpass.getpass("Authy backup password: ")

    # utf-8-sig drops a leading BOM, which would otherwise become part of
    # the first column name and break header detection.
    content = normalize_authy_csv(args.input.read_text(encoding="utf-8-sig"))

    reader = csv.reader(io.StringIO(content))
    header = next(reader, [])
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        print(
            f"Error: CSV is missing required columns: {sorted(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    successes = []
    failures = []
    malformed = 0

    for fields in reader:
        if not any(f.strip() for f in fields):
            continue
        try:
            row = parse_row(header, fields)
        except ValueError as exc:
            # Report by line number: the fields may not be where we expect,
            # so any of them could hold key material.
            malformed += 1
            print(f"  ✗  line {reader.line_num}: {exc}")
            continue

        name = row["name"].strip()
        enc_seed = row["encrypted_seed"].strip()
        salt = row["salt"].strip()
        iv = row["iv"].strip()

        if not name or not enc_seed:
            continue

        try:
            secret = decrypt_seed(enc_seed, salt, iv, password)
            issuer, account = split_name(name)
            successes.append(
                {"name": name, "issuer": issuer, "account": account, "secret": secret}
            )
            print(f"  ✓  {name}")
        except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
            failures.append((name, exc))
            print(f"  ✗  {name}: {exc}")

    problems = []
    if failures:
        problems.append(f"{len(failures)} failed to decrypt")
    if malformed:
        problems.append(f"{malformed} malformed row(s)")

    if not successes and not problems:
        print("\nNo valid token rows found in CSV.", file=sys.stderr)
        sys.exit(1)
    if not successes:
        # Only blame the password when rows were well-formed but still failed.
        hint = " — wrong backup password?" if failures else ""
        print(f"\nNo tokens decrypted ({', '.join(problems)}){hint}", file=sys.stderr)
        sys.exit(1)

    written = []

    if args.format in ("aegis", "both"):
        aegis_path = output_dir / args.aegis_file
        write_secret_file(aegis_path, json.dumps(build_aegis(successes), indent=2))
        written.append(aegis_path)

    if args.format in ("uris", "both"):
        uris_path = output_dir / args.uris_file
        uris = "".join(
            build_otpauth(t["issuer"], t["account"], t["secret"]) + "\n"
            for t in successes
        )
        write_secret_file(uris_path, uris)
        written.append(uris_path)

    print(f"\nDecrypted {len(successes)} token(s)" + "".join(f", {p}" for p in problems))
    print("\nOutput:")
    for path in written:
        print(f"  {path}")
    print("\n⚠  Delete these files after importing — they contain plaintext secrets.")


if __name__ == "__main__":
    main()
