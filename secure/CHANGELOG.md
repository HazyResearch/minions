# Changelog

All notable changes to the secure minions module will be documented in this file. 

## [Latest] - 2025-06-09

### Added
- VM measurement hash for enhanced attestation verification (at the VM level)

## [2025-06-08]

### Added
- Pinned attestation logic for improved security
- CPU attestation support
- Secure GPU attestation verification via NVIDIA Attestation Tools

### Fixed
- GPU attestation implementation improvements

### Changed
- Updated documentation with security caveats

## [2025-06-07]

### Added
- SNP Azure attestation and documentation
- Manual attestation key pinning
- TLS certificate binding to GPU attestation report

### Fixed
- Secure documentation updates

## [2025-06-05]

### Added
- SSL wrapper to secure server to ensure HTTPS
- HTTPS configuration documentation
- Protection against non-HTTPS supervisor client URLs

### Changed
- Updated secure README with HTTPS configuration details

## [2025-06-01]

### Changed
- Updated README for secure minions

## [2025-05-29]

### Added
- Secure minions functionality (local llm - remote llm).

## [2025-05-18]

### Added
- New utilities file for chat processing
- Enhanced server logging

### Changed
- Code reformatting and cleanup

## [2025-05-17]

### Changed
- Updated secure README documentation

## [2025-05-16]

### Added
- Folder upload capabilities to secure chat
- PDF upload functionality

### Fixed
- Removed unnecessary imports

### Changed
- Updated README documentation

## [2025-05-14]

### Added
- PDF upload support to secure chat application

## [2025-05-12]

### Added
- Image support for secure chat
- Server updates and improvements

### Changed
- Repository cleanup
- Updated README documentation

## [2025-05-11]

### Added
- Enhanced attestation verification system
- Secure chat server details

### Changed
- Updated README with attestation information

## [2025-05-10]

### Added
- Multimodal support for secure chat
- Image support in secure chat protocol
- Initial README documentation

### Fixed
- Import statements and dependencies

## [2025-05-09]

### Added
- Initial secure chat protocol and application
- Attestation verification system
- Crypto utilities for encryption and signing
- Worker server + local caht client
- Support for multiple AI model providers:
  - Anthropic
  - Azure OpenAI
  - Cartesia MLX
  - DeepSeek
  - Gemini
  - Groq
  - Hugging Face
  - MLX Audio/LM/Omni
  - Ollama
  - OpenAI
  - OpenRouter
  - Perplexity
  - SambaNova
  - Together AI
  - Tokasaurus
- Minion prompt templates
