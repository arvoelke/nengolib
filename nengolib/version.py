"""Nengo Library version information."""

version_info = (0, 4, 0)  # (major, minor, patch)
release_type = "-beta"  # or "-dev" or ""

version = "%s%s" % (".".join(map(str, version_info)), release_type)
