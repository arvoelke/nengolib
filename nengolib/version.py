"""Nengo Library version information."""

version_info = (0, 4, 0)  # (major, minor, patch)
dev = False

version = "%s%s" % (".".join(map(str, version_info)), "-dev" if dev else "")
