import base64
import csv
import hashlib
import io
import json
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import authy_migrate
from authy_migrate import (
    KDF_ITERATION_CANDIDATES,
    KDF_ITERATIONS,
    build_aegis,
    build_bitwarden,
    build_otpauth,
    decrypt_seed,
    decrypt_seed_with_iterations,
    main,
    normalize_authy_csv,
    parse_row,
    split_name,
    write_secret_file,
)

# ---------------------------------------------------------------------------
# Shared test vectors
# ---------------------------------------------------------------------------

_PASSWORD = "hunter2"
_SALT = "somesalt"
_IV_HEX = "deadbeefdeadbeefdeadbeefdeadbeef"
_SECRET = "JBSWY3DPEHPK3PXP"


def _encrypt(
    secret: str,
    password: str = _PASSWORD,
    salt: str = _SALT,
    iv_hex: str = _IV_HEX,
    iterations: int = 100_000,
) -> str:
    """Encrypt a secret the same way Authy does, for use in test fixtures."""
    key = hashlib.pbkdf2_hmac(
        "sha1", password.encode(), salt.encode(), iterations, dklen=32
    )
    return _encrypt_plaintext(secret.encode("utf-8"), key, iv_hex)


def _encrypt_plaintext(plain: bytes, key: bytes, iv_hex: str = _IV_HEX) -> str:
    """PKCS#7-pad and AES-256-CBC encrypt arbitrary bytes under a given key."""
    pad_len = 16 - (len(plain) % 16)
    padded = plain + bytes([pad_len] * pad_len)
    return _encrypt_padded(padded, key, iv_hex)


def _encrypt_padded(padded: bytes, key: bytes, iv_hex: str = _IV_HEX) -> str:
    """AES-256-CBC encrypt already-padded bytes under a given key."""
    encryptor = Cipher(algorithms.AES(key), modes.CBC(bytes.fromhex(iv_hex))).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode()


_ENCRYPTED = _encrypt(_SECRET)
_ENCRYPTED_LEGACY = _encrypt(_SECRET, iterations=1_000)


def _make_csv(rows: list[dict], wrap: bool = False, wrap_lines: bool = False) -> str:
    """Build an Authy-formatted CSV string.

    wrap:       older Authy exports wrap the entire file in one pair of quotes.
    wrap_lines: newer Authy exports wrap each line in its own pair of quotes.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["name", "encrypted_seed", "salt", "iv"])
    writer.writeheader()
    writer.writerows(rows)
    content = buf.getvalue()
    if wrap_lines:
        return "".join(f'"{line}"\n' for line in content.splitlines())
    return f'"{content}"' if wrap else content


def _make_unquoted_csv(rows: list[dict], wrapping: str = "plain") -> str:
    """Like _make_csv, but with no field quoting, as Authy writes names containing commas.

    wrapping: "plain", "whole_file" or "per_line", matching the Authy formats.
    """
    content = _make_csv(rows).replace('"', "")
    if wrapping == "per_line":
        return "".join(f'"{line}"\n' for line in content.splitlines())
    return f'"{content}"' if wrapping == "whole_file" else content


def _token_row(
    name: str = "GitHub:user@example.com", encrypted_seed: str = _ENCRYPTED
) -> dict:
    return {"name": name, "encrypted_seed": encrypted_seed, "salt": _SALT, "iv": _IV_HEX}


# ---------------------------------------------------------------------------
# decrypt_seed
# ---------------------------------------------------------------------------


class TestDecryptSeed:
    def test_correct_password_returns_secret(self):
        assert decrypt_seed(_ENCRYPTED, _SALT, _IV_HEX, _PASSWORD) == _SECRET

    def test_result_is_uppercase(self):
        enc = _encrypt(_SECRET.lower())
        assert decrypt_seed(enc, _SALT, _IV_HEX, _PASSWORD) == _SECRET.upper()

    def test_wrong_password_raises(self):
        with pytest.raises((ValueError, UnicodeDecodeError)):
            decrypt_seed(_ENCRYPTED, _SALT, _IV_HEX, "wrong")

    def test_wrong_salt_raises(self):
        with pytest.raises((ValueError, UnicodeDecodeError)):
            decrypt_seed(_ENCRYPTED, "badsalt", _IV_HEX, _PASSWORD)

    def test_wrong_iv_does_not_yield_original_secret(self):
        # With wrong IV the first plaintext block is garbled, so the result either
        # fails to decode or differs from the original — never matches.
        bad_iv = "00" * 16
        try:
            result = decrypt_seed(_ENCRYPTED, _SALT, bad_iv, _PASSWORD)
        except (ValueError, UnicodeDecodeError):
            return
        assert result != _SECRET

    def test_bad_padding_error_does_not_leak_data(self):
        # Correct key and IV, so the secret itself decrypts, but the final
        # block is all zero bytes, which is not valid PKCS#7 padding.
        key = hashlib.pbkdf2_hmac(
            "sha1", _PASSWORD.encode(), _SALT.encode(), 100_000, dklen=32
        )
        enc = _encrypt_padded(_SECRET.encode("utf-8") + bytes(16), key)
        ciphertext = base64.b64decode(enc)
        with pytest.raises(ValueError) as exc:
            decrypt_seed(enc, _SALT, _IV_HEX, _PASSWORD)
        message = str(exc.value)
        assert "Invalid padding" in message
        for leaked in (_SECRET, _SECRET.lower(), enc, ciphertext.hex(), repr(ciphertext)):
            assert leaked not in message


class TestKdfIterationFallback:
    def test_candidates_start_with_default(self):
        assert KDF_ITERATION_CANDIDATES == (100_000, 1_000)
        assert KDF_ITERATIONS == KDF_ITERATION_CANDIDATES[0]

    def test_legacy_1000_round_row_decrypts(self):
        assert decrypt_seed_with_iterations(
            _ENCRYPTED_LEGACY, _SALT, _IV_HEX, _PASSWORD
        ) == (_SECRET, 1_000)
        assert decrypt_seed(_ENCRYPTED_LEGACY, _SALT, _IV_HEX, _PASSWORD) == _SECRET

    def test_default_100000_round_row_is_tried_first(self):
        real_pbkdf2 = hashlib.pbkdf2_hmac
        attempted = []

        def recording_pbkdf2(hash_name, password, salt, iterations, dklen=None):
            attempted.append(iterations)
            return real_pbkdf2(hash_name, password, salt, iterations, dklen)

        with patch.object(authy_migrate.hashlib, "pbkdf2_hmac", recording_pbkdf2):
            result = decrypt_seed_with_iterations(_ENCRYPTED, _SALT, _IV_HEX, _PASSWORD)

        assert result == (_SECRET, 100_000)
        assert attempted == [100_000]

    def test_wrong_password_fails_for_every_candidate(self):
        real_pbkdf2 = hashlib.pbkdf2_hmac
        attempted = []

        def recording_pbkdf2(hash_name, password, salt, iterations, dklen=None):
            attempted.append(iterations)
            return real_pbkdf2(hash_name, password, salt, iterations, dklen)

        for encrypted in (_ENCRYPTED, _ENCRYPTED_LEGACY):
            attempted.clear()
            with patch.object(authy_migrate.hashlib, "pbkdf2_hmac", recording_pbkdf2):
                with pytest.raises(ValueError, match="wrong password"):
                    decrypt_seed_with_iterations(encrypted, _SALT, _IV_HEX, "wrong")
            assert attempted == list(KDF_ITERATION_CANDIDATES)

    @pytest.mark.parametrize(
        "plaintext",
        [
            b"not base32 at all!",
            b"JBSWY3DP0189",  # digits 0, 1, 8 and 9 are outside the base32 alphabet
            b"JBSW=Y3DP",  # padding only allowed at the end
            b"   ",  # empty after strip
            b"\xff\xfe\xfd",  # not UTF-8
        ],
    )
    def test_padding_valid_but_non_base32_plaintext_is_rejected(self, plaintext):
        for iterations in KDF_ITERATION_CANDIDATES:
            key = hashlib.pbkdf2_hmac(
                "sha1", _PASSWORD.encode(), _SALT.encode(), iterations, dklen=32
            )
            encrypted = _encrypt_plaintext(plaintext, key)
            with pytest.raises(ValueError, match="wrong password") as exc:
                decrypt_seed_with_iterations(encrypted, _SALT, _IV_HEX, _PASSWORD)
            assert type(exc.value) is ValueError

    def test_error_message_does_not_leak_inputs(self):
        with pytest.raises(ValueError) as exc:
            decrypt_seed(_ENCRYPTED, _SALT, _IV_HEX, "not-the-password")
        message = str(exc.value)
        for value in (_ENCRYPTED, _SALT, _IV_HEX, "not-the-password", _SECRET):
            assert value not in message

    def test_base32_padding_is_accepted(self):
        enc = _encrypt("JBSWY3DPEE======")
        assert decrypt_seed(enc, _SALT, _IV_HEX, _PASSWORD) == "JBSWY3DPEE======"


# ---------------------------------------------------------------------------
# normalize_authy_csv
# ---------------------------------------------------------------------------


class TestNormalizeAuthyCsv:
    PLAIN = "name,encrypted_seed,salt,iv\nGitHub:me,AAA,BBB,CCC"

    def test_plain_csv_is_unchanged(self):
        assert normalize_authy_csv(self.PLAIN) == self.PLAIN

    def test_strips_whole_file_quotes(self):
        assert normalize_authy_csv(f'"{self.PLAIN}"\n') == self.PLAIN

    def test_strips_per_line_quotes(self):
        quoted = '"name,encrypted_seed,salt,iv"\n"GitHub:me,AAA,BBB,CCC"\n'
        assert normalize_authy_csv(quoted) == self.PLAIN

    def test_leaves_legitimately_quoted_field_alone(self):
        # A name containing a comma is quoted by the CSV writer; the line does
        # not end with a quote, so it must pass through untouched.
        line = '"Foo, Inc:me",AAA,BBB,CCC'
        assert normalize_authy_csv(f"name,encrypted_seed,salt,iv\n{line}") == (
            f"name,encrypted_seed,salt,iv\n{line}"
        )

    def test_lone_quote_is_not_stripped(self):
        assert normalize_authy_csv('"') == '"'

    @pytest.mark.parametrize("quoted_header", [False, True])
    @pytest.mark.parametrize("wrapping", ["plain", "whole_file", "per_line"])
    def test_preserves_field_quotes(self, quoted_header, wrapping):
        header = ('"name","encrypted_seed","salt","iv"' if quoted_header
                  else "name,encrypted_seed,salt,iv")
        plain = header + '\n"Foo, Inc:me","AAA","BBB","CCC"'
        if wrapping == "whole_file":
            content = f'"{plain}"'
        elif wrapping == "per_line":
            content = "\n".join(f'"{line}"' for line in plain.splitlines())
        else:
            content = plain
        rows = list(csv.DictReader(io.StringIO(normalize_authy_csv(content))))
        assert rows == [{"name": "Foo, Inc:me", "encrypted_seed": "AAA",
                         "salt": "BBB", "iv": "CCC"}]

    def test_whole_file_wrapper_with_quoted_final_name_without_newline(self):
        plain = 'name,encrypted_seed,salt,iv\n"Foo, Inc:me",AAA,BBB,CCC'
        assert normalize_authy_csv(f'"{plain}"') == plain


# ---------------------------------------------------------------------------
# parse_row
# ---------------------------------------------------------------------------


class TestParseRow:
    HEADER = ["name", "encrypted_seed", "salt", "iv"]

    def test_exact_field_count(self):
        assert parse_row(self.HEADER, ["GitHub:me", "AAA", "BBB", "CCC"]) == {
            "name": "GitHub:me", "encrypted_seed": "AAA", "salt": "BBB", "iv": "CCC"
        }

    def test_surplus_fields_are_joined_into_name(self):
        fields = ["Acme:Surname", "Firstname", " Jr", "AAA", "BBB", "CCC"]
        assert parse_row(self.HEADER, fields) == {
            "name": "Acme:Surname,Firstname, Jr",
            "encrypted_seed": "AAA",
            "salt": "BBB",
            "iv": "CCC",
        }

    def test_surplus_fields_left_alone_for_unknown_column_order(self):
        header = ["encrypted_seed", "salt", "iv", "name"]
        assert parse_row(header, ["AAA", "BBB", "CCC", "Acme:Surname", "Firstname"]) == {
            "encrypted_seed": "AAA", "salt": "BBB", "iv": "CCC", "name": "Acme:Surname"
        }

    def test_too_few_fields_raises_without_field_contents(self):
        with pytest.raises(ValueError) as exc:
            parse_row(self.HEADER, ["Broken:row", "AAA"])
        assert str(exc.value) == "malformed row: expected 4 fields, got 2"


# ---------------------------------------------------------------------------
# split_name
# ---------------------------------------------------------------------------


class TestSplitName:
    def test_service_and_account(self):
        assert split_name("GitHub:user@example.com") == ("GitHub", "user@example.com")

    def test_no_colon_gives_empty_account(self):
        assert split_name("GitHub") == ("GitHub", "")

    def test_strips_whitespace(self):
        assert split_name("  GitHub : user  ") == ("GitHub", "user")

    def test_only_first_colon_is_split(self):
        assert split_name("Service:user:extra") == ("Service", "user:extra")


# ---------------------------------------------------------------------------
# build_otpauth
# ---------------------------------------------------------------------------


class TestBuildOtpauth:
    def test_with_account(self):
        uri = build_otpauth("GitHub", "user@example.com", "ABCDEF")
        # '@' is a reserved URI char and must be percent-encoded
        assert uri.startswith("otpauth://totp/GitHub:user%40example.com?")
        assert "secret=ABCDEF" in uri
        assert "issuer=GitHub" in uri

    def test_without_account_omits_colon(self):
        uri = build_otpauth("GitHub", "", "ABCDEF")
        assert "otpauth://totp/GitHub?" in uri
        assert "GitHub:?" not in uri

    def test_fixed_parameters(self):
        uri = build_otpauth("X", "y", "Z")
        assert "algorithm=SHA1" in uri
        assert "digits=6" in uri
        assert "period=30" in uri

    def test_special_characters_are_url_encoded(self):
        uri = build_otpauth("Acme & Co", "user?id=1", "ABCDEF")
        # raw '&' and '?' must not appear inside the label or issuer
        assert "Acme & Co" not in uri
        assert "user?id=1" not in uri
        assert "Acme%20%26%20Co" in uri
        assert "user%3Fid%3D1" in uri
        # secret stays unencoded so importers can read it
        assert "secret=ABCDEF" in uri

    def test_slash_in_issuer_is_encoded(self):
        uri = build_otpauth("Mail/Cloud", "a", "S")
        assert "Mail/Cloud" not in uri
        assert "Mail%2FCloud" in uri


# ---------------------------------------------------------------------------
# build_aegis
# ---------------------------------------------------------------------------


class TestBuildAegis:
    def test_top_level_structure(self):
        result = build_aegis([{"issuer": "X", "account": "a", "secret": "S"}])
        assert result["version"] == 1
        assert result["db"]["version"] == 3
        assert result["db"]["groups"] == []

    def test_entry_has_all_fields_required_by_strict_importers(self):
        # Proton Authenticator deserializes Aegis entries into a struct with no
        # defaults, so a missing key (notably "groups") rejects the whole file.
        entry = build_aegis([{"issuer": "X", "account": "a", "secret": "S"}])["db"][
            "entries"
        ][0]
        assert set(entry) == {
            "type", "uuid", "name", "issuer", "note", "favorite", "icon", "info", "groups"
        }
        assert entry["groups"] == []
        assert entry["note"] == ""
        assert entry["favorite"] is False
        assert entry["icon"] is None

    def test_entry_fields(self):
        entry = build_aegis(
            [{"issuer": "GitHub", "account": "user", "secret": "ABCDEF"}]
        )["db"]["entries"][0]
        assert entry["type"] == "totp"
        assert entry["issuer"] == "GitHub"
        assert entry["name"] == "user"
        assert entry["info"] == {
            "secret": "ABCDEF",
            "algo": "SHA1",
            "digits": 6,
            "period": 30,
        }

    def test_name_falls_back_to_issuer_when_no_account(self):
        entry = build_aegis([{"issuer": "GitHub", "account": "", "secret": "X"}])["db"][
            "entries"
        ][0]
        assert entry["name"] == "GitHub"

    def test_entry_count_matches_input(self):
        tokens = [
            {"issuer": "A", "account": "a", "secret": "X"},
            {"issuer": "B", "account": "b", "secret": "Y"},
        ]
        entries = build_aegis(tokens)["db"]["entries"]
        assert len(entries) == 2

    def test_each_entry_has_unique_uuid(self):
        tokens = [
            {"issuer": "A", "account": "a", "secret": "X"},
            {"issuer": "B", "account": "b", "secret": "Y"},
        ]
        uuids = [e["uuid"] for e in build_aegis(tokens)["db"]["entries"]]
        assert len(set(uuids)) == 2


# ---------------------------------------------------------------------------
# build_bitwarden
# ---------------------------------------------------------------------------


def _bw_token(name: str, secret: str = "ABCDEF") -> dict:
    issuer, account = split_name(name)
    return {"name": name, "issuer": issuer, "account": account, "secret": secret}


class TestBuildBitwarden:
    def test_top_level_structure(self):
        result = build_bitwarden([_bw_token("X:a")])
        assert result["encrypted"] is False
        assert result["folders"] == []
        assert len(result["items"]) == 1

    def test_item_with_issuer_and_account(self):
        item = build_bitwarden([_bw_token("GitHub:user@example.com")])["items"][0]
        assert set(item) == {
            "id", "type", "name", "favorite", "notes", "login", "folderId", "collectionIds"
        }
        assert item["type"] == 1
        assert item["name"] == "GitHub"
        assert item["favorite"] is False
        assert item["notes"] is None
        assert item["folderId"] is None
        assert item["collectionIds"] is None
        assert item["login"] == {
            "username": "user@example.com",
            "password": None,
            "totp": build_otpauth("GitHub", "user@example.com", "ABCDEF"),
            "uris": [],
        }

    def test_empty_account_gives_null_username(self):
        item = build_bitwarden([_bw_token("GitHub:")])["items"][0]
        assert item["name"] == "GitHub"
        assert item["login"]["username"] is None
        assert item["login"]["totp"] == build_otpauth("GitHub", "", "ABCDEF")

    def test_name_without_colon_uses_raw_name(self):
        item = build_bitwarden([_bw_token("GitHub")])["items"][0]
        assert item["name"] == "GitHub"
        assert item["login"]["username"] is None

    def test_empty_issuer_falls_back_to_raw_name(self):
        item = build_bitwarden([_bw_token(":user")])["items"][0]
        assert item["name"] == ":user"
        assert item["login"]["username"] == "user"

    def test_items_have_distinct_ids_and_no_password(self):
        items = build_bitwarden(
            [_bw_token("A:a", "X"), _bw_token("B:b", "Y"), _bw_token("C", "Z")]
        )["items"]
        assert len({i["id"] for i in items}) == 3
        assert all(i["login"]["password"] is None for i in items)


# ---------------------------------------------------------------------------
# write_secret_file
# ---------------------------------------------------------------------------


class TestWriteSecretFile:
    def test_content_is_written(self, tmp_path):
        path = tmp_path / "out.txt"
        write_secret_file(path, "hello")
        assert path.read_text() == "hello"

    def test_new_file_has_0600_permissions(self, tmp_path):
        path = tmp_path / "out.txt"
        write_secret_file(path, "secret")
        assert (path.stat().st_mode & 0o777) == 0o600

    def test_preexisting_loose_permissions_are_tightened(self, tmp_path):
        path = tmp_path / "out.txt"
        path.write_text("old")
        path.chmod(0o644)
        write_secret_file(path, "new")
        assert path.read_text() == "new"
        assert (path.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# Integration: main()
# ---------------------------------------------------------------------------


def _run(args: list[str], password: str = _PASSWORD):
    """Call main() with patched argv and getpass."""
    with patch("sys.argv", ["authy_migrate.py"] + args), patch(
        "getpass.getpass", return_value=password
    ):
        main()


class TestMain:
    def test_decrypts_token_to_both_formats(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file)])

        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert aegis["db"]["entries"][0]["info"]["secret"] == _SECRET

        uris = (tmp_path / "otpauth_uris.txt").read_text()
        assert f"secret={_SECRET}" in uris

    def test_handles_authy_quoted_csv_wrapping(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()], wrap=True))
        _run([str(csv_file)])
        assert (tmp_path / "tokens_aegis.json").exists()

    def test_handles_authy_per_line_quoted_csv(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()], wrap_lines=True))
        _run([str(csv_file)])

        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert aegis["db"]["entries"][0]["info"]["secret"] == _SECRET

    def test_handles_utf8_bom(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()], wrap_lines=True), encoding="utf-8-sig")
        _run([str(csv_file)])

        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert aegis["db"]["entries"][0]["info"]["secret"] == _SECRET

    def test_output_files_have_0600_permissions(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file)])
        for name in ("tokens_aegis.json", "otpauth_uris.txt"):
            assert (tmp_path / name).stat().st_mode & 0o777 == 0o600

    def test_output_dir_option(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        out = tmp_path / "out"
        out.mkdir()
        _run([str(csv_file), "-o", str(out)])
        assert (out / "tokens_aegis.json").exists()

    def test_format_aegis_only(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "aegis"])
        assert (tmp_path / "tokens_aegis.json").exists()
        assert not (tmp_path / "otpauth_uris.txt").exists()

    def test_format_uris_only(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "uris"])
        assert not (tmp_path / "tokens_aegis.json").exists()
        assert (tmp_path / "otpauth_uris.txt").exists()

    def test_format_bitwarden_only(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "bitwarden"])
        path = tmp_path / "tokens_bitwarden.json"
        assert path.stat().st_mode & 0o777 == 0o600
        bitwarden = json.loads(path.read_text())
        assert bitwarden["items"][0]["login"]["totp"] == build_otpauth(
            "GitHub", "user@example.com", _SECRET
        )
        assert not (tmp_path / "tokens_aegis.json").exists()
        assert not (tmp_path / "otpauth_uris.txt").exists()

    def test_bitwarden_file_option(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "bitwarden", "--bitwarden-file", "bw.json"])
        assert (tmp_path / "bw.json").exists()
        assert not (tmp_path / "tokens_bitwarden.json").exists()

    def test_format_both_excludes_bitwarden(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "both"])
        assert (tmp_path / "tokens_aegis.json").exists()
        assert (tmp_path / "otpauth_uris.txt").exists()
        assert not (tmp_path / "tokens_bitwarden.json").exists()

    def test_format_all_writes_every_file(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file), "--format", "all"])
        for name in ("tokens_aegis.json", "otpauth_uris.txt", "tokens_bitwarden.json"):
            assert (tmp_path / name).stat().st_mode & 0o777 == 0o600

    def test_exits_1_on_missing_input_file(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _run([str(tmp_path / "ghost.csv")])
        assert exc.value.code == 1

    def test_exits_1_on_missing_csv_columns(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text("wrong,columns\nfoo,bar\n")
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file)])
        assert exc.value.code == 1

    def test_exits_1_on_empty_csv(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([]))
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file)])
        assert exc.value.code == 1

    def test_exits_1_on_wrong_password(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file)], password="wrong")
        assert exc.value.code == 1

    def test_exits_1_on_nonexistent_output_dir(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file), "-o", str(tmp_path / "ghost")])
        assert exc.value.code == 1

    def test_skips_rows_with_empty_name(self, tmp_path):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(
            _make_csv(
                [
                    {
                        "name": "",
                        "encrypted_seed": _ENCRYPTED,
                        "salt": _SALT,
                        "iv": _IV_HEX,
                    },
                    _token_row("Valid:user"),
                ]
            )
        )
        _run([str(csv_file)])
        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert len(aegis["db"]["entries"]) == 1

    @pytest.mark.parametrize("wrapping", ["plain", "whole_file", "per_line"])
    def test_unquoted_comma_in_name_is_preserved(self, tmp_path, wrapping):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(
            _make_unquoted_csv([_token_row("Acme:Surname,Firstname")], wrapping)
        )
        _run([str(csv_file)])

        entry = json.loads((tmp_path / "tokens_aegis.json").read_text())["db"][
            "entries"
        ][0]
        assert entry["issuer"] == "Acme"
        assert entry["name"] == "Surname,Firstname"
        assert entry["info"]["secret"] == _SECRET

        uris = (tmp_path / "otpauth_uris.txt").read_text()
        assert "otpauth://totp/Acme:Surname%2CFirstname?" in uris

    def test_row_with_too_few_fields_is_reported_as_failure(self, tmp_path, capsys):
        csv_file = tmp_path / "tokens.csv"
        good = _make_csv([_token_row("First:user"), _token_row("Second:user")])
        header, first, second = good.splitlines()
        csv_file.write_text(f"{header}\n{first}\nBroken:row,PLACEHOLDER\n{second}\n")
        _run([str(csv_file)])

        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert [e["issuer"] for e in aegis["db"]["entries"]] == ["First", "Second"]

        out = capsys.readouterr().out
        assert "✗  line 3: malformed row: expected 4 fields, got 2" in out
        assert "Decrypted 2 token(s), 1 malformed row(s)" in out
        assert "failed" not in out
        assert "Broken" not in out
        assert "PLACEHOLDER" not in out

    def test_summary_counts_decrypt_failures_and_malformed_rows_separately(
        self, tmp_path, capsys
    ):
        csv_file = tmp_path / "tokens.csv"
        bad_iv = {**_token_row("BadIv:user"), "iv": "not-hex"}
        good = _make_csv([_token_row("Good:user"), bad_iv])
        csv_file.write_text(good + "Broken:row,PLACEHOLDER\n")
        _run([str(csv_file)])

        out = capsys.readouterr().out
        assert "Decrypted 1 token(s), 1 failed to decrypt, 1 malformed row(s)" in out

    def test_all_rows_malformed_does_not_blame_password(self, tmp_path, capsys):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text("name,encrypted_seed,salt,iv\nBroken:row,PLACEHOLDER\n")
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file)])
        assert exc.value.code == 1

        err = capsys.readouterr().err
        assert "No tokens decrypted (1 malformed row(s))" in err
        assert "password" not in err

    def test_wrong_password_with_malformed_rows_reports_both(self, tmp_path, capsys):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]) + "Broken:row,PLACEHOLDER\n")
        with pytest.raises(SystemExit) as exc:
            _run([str(csv_file)], password="wrong")
        assert exc.value.code == 1

        err = capsys.readouterr().err
        assert (
            "No tokens decrypted (1 failed to decrypt, 1 malformed row(s))"
            " — wrong backup password?"
        ) in err
    def test_legacy_iteration_summary_shown_when_needed(self, tmp_path, capsys):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(
            _make_csv(
                [
                    _token_row("New:user"),
                    _token_row("Old:one", encrypted_seed=_ENCRYPTED_LEGACY),
                    _token_row("Old:two", encrypted_seed=_ENCRYPTED_LEGACY),
                ]
            )
        )
        _run([str(csv_file)])
        out = capsys.readouterr().out
        assert "2 token(s) used the legacy 1,000-round key derivation" in out
        for name in ("New:user", "Old:one", "Old:two"):
            assert f"  ✓  {name}\n" in out

        aegis = json.loads((tmp_path / "tokens_aegis.json").read_text())
        assert [e["info"]["secret"] for e in aegis["db"]["entries"]] == [_SECRET] * 3

    def test_legacy_iteration_summary_absent_by_default(self, tmp_path, capsys):
        csv_file = tmp_path / "tokens.csv"
        csv_file.write_text(_make_csv([_token_row()]))
        _run([str(csv_file)])
        out = capsys.readouterr().out
        assert "key derivation" not in out
