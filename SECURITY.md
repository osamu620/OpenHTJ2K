# Security policy

Please report suspected vulnerabilities privately through GitHub's
**[Report a vulnerability](https://github.com/osamu620/OpenHTJ2K/security/advisories/new)**
workflow. Do not open a public issue or pull request for a security
issue that has not already been publicly disclosed.

## Supported versions

Only the most recent released tag is supported for security fixes.
Older tags are end of life — upgrade rather than request a backport.

| Version              | Status      |
| -------------------- | ----------- |
| Latest released tag  | Supported   |
| Anything older       | End of life |

## What to include

A reproducible report is much faster to triage. If you can include:

- Affected commit (`git rev-parse HEAD`) and build mode
  (`CMAKE_BUILD_TYPE`, SIMD dispatch, thread pool on/off).
- A minimal input file and the exact command line that reproduces the
  issue, ideally against an AddressSanitizer-instrumented build.
- Observed vs expected behaviour.
- Root cause analysis if you already have it.

File sizes up to a few MB are fine as GitHub attachments; larger
corpora, drop a link (Dropbox, S3, GCS signed URL, etc.).

## What to expect

- An acknowledgement within **3 working days** of a report being filed.
- A fix or a mitigation plan within **14 days** for most issues;
  complex crashes affecting the public decoder API are prioritised.
- Default disclosure posture is **fix-forward publicly**: we'll
  coordinate a private fix branch only when the root cause is subtle
  enough that a public patch would hand attackers a recipe. Reports
  whose root cause and fix are already obvious (e.g. a bounds-check
  omission) are typically landed publicly with full credit.

## CVE coordination

Mention in your report if you'd like a CVE assigned and I'll open a
GitHub security advisory with you listed as the reporter — that's
the path that gets the CVE issued. If you have a different CVE
channel you'd rather use (MITRE direct, your organisation's CNA,
etc.), tell me in the report and we'll figure it out together.

## Credit

Reporters are credited by the exact name, affiliation, and (if
supplied) GitHub handle they provide in the report. Both the
GitHub Security Advisory and the `CHANGELOG` entry for the
patch-level release carry the credit line.

## Scope

**In scope**
- Memory safety issues (out-of-bounds reads/writes, use-after-free,
  stack overflow, heap overflow, uninitialized reads, integer
  overflow leading to any of the above) in the decoder or encoder.
- Crashes or hangs reachable from a malformed but well-formed-ish
  JPEG 2000 / HTJ2K codestream.
- Issues in the JPIP server or parser reachable from a crafted
  HTTP request or JPP-stream response.

**Out of scope**
- Crashes from inputs that exceed documented format limits (e.g. a
  canvas size that blows the configured memory budget). These are
  parameter-validation bugs at worst and should be filed as regular
  issues.
- Denial of service purely via resource exhaustion on well-formed
  but oversized inputs.
- Any issue requiring modification of the library's source at build
  time.
