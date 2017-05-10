"""Nengo Library version information."""

version_info = (0, 2, 0)  # (major, minor, patch)
dev = True

version = "%s%s" % (".".join(map(str, version_info)), "-dev" if dev else "")
