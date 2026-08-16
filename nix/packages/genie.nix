{ lib, rustPlatform, fetchFromGitHub }:

rustPlatform.buildRustPackage rec {
  pname = "genie";
  version = "0.1.0";
  src = ./../../genie;
  cargoLock.lockFile = ./../../genie/Cargo.lock;

  meta = with lib; {
    description = "1-bit multimodal liquid diffusion LLM";
    homepage = "https://github.com/falconnor4/Lamp";
    license = licenses.mit;
  };
}