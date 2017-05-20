"""Nengo Library version information."""

version_info = (0, 3, 0)  # (major, minor, patch)
dev = True

version = "%s%s" % (".".join(map(str, version_info)), "-dev" if dev else "")
