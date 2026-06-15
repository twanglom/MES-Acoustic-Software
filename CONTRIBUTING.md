# Contributing

Contributions are welcome through issues and pull requests.

## Reporting A Bug

Include:

- MES-Acoustic version
- Windows version
- Whether the application was installed or run from source
- Steps to reproduce the problem
- The complete error message
- Mesh cell count and input format
- A small non-confidential test case when possible

Do not upload proprietary geometry or large view-factor matrices publicly.

## Pull Requests

1. Create a focused branch.
2. Keep changes limited to one problem or feature.
3. Follow the existing PySide6 and `mes.ret` structure.
4. Add or update documentation for user-visible behavior.
5. Test mesh import, Physical Surface assignment, VF handling, and solver
   execution when those areas are affected.
6. Describe verification performed in the pull request.

## Generated Files

Do not commit build output, local case data, project output, or generated
view-factor matrices. The repository `.gitignore` excludes the standard
locations.
