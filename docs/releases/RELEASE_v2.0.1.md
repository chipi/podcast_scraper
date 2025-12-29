# Release v2.0.1 - Legal Notices and Usage Clarifications

**Release Date:** November 13, 2025
**Type:** Patch Release

## Summary

v2.0.1 is a patch release that adds comprehensive legal notices and usage clarifications to the project. This release addresses issue #25 by clearly distinguishing between code licensing (MIT) and content licensing (subject to original creator's copyright), and provides clear guidance on intended use and user responsibilities.

## What's New

### Legal Documentation

- **LICENSE file**: Added note clarifying that MIT license applies only to source code, not downloaded podcast content
- **docs/LEGAL.md**: Created comprehensive legal notice document covering:
  - Overview and intended use (personal, non-commercial)
  - What users may do (personal use)
  - What users may not do (redistribution, commercial use)
  - Code license vs. content rights
  - User responsibilities
  - Maintainer intent

### README Updates

- Added "Personal Use Only" badge next to License badge
- Added early warning notice immediately after project description
- Added usage reminder in Usage section
- Added "Project Intent & Fair Use Notice" section with detailed guidelines
- Cross-referenced LEGAL.md throughout

### Package Updates

- \***\*init**.py\*\*: Added header comment with legal notice
- **mkdocs.yml**: Added Legal Notice to documentation navigation (as last item)

## Key Points

1. **Code Licensing**: MIT License applies to the `podcast_scraper` codebase
2. **Content Licensing**: Downloaded transcripts/content remain subject to original creator's copyright
3. **Intended Use**: Personal, non-commercial use only
4. **User Responsibility**: Users must ensure compliance with copyright law, RSS feed terms, and platform policies
5. **No Redistribution**: All downloaded content must remain local and cannot be shared or redistributed

## Related Issues

- Closes #25: Clarify usage intent and licensing (code vs content)

## Documentation

- Legal Notice: [docs/LEGAL.md](../../LEGAL.md)
- Project Intent: See "Project Intent & Fair Use Notice" section in README.md

## Migration Notes

No code changes in this release - only documentation and legal notices added. No migration required.

## Contributors

- Marko Dragoljevic (@chipi)
