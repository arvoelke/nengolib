"""Nengo Library version information."""

version_info = (0, 1, 0)  # (major, minor, patch)
dev = True

version = "%s%s" % (".".join(map(str, version_info)), "-dev" if dev else "")
