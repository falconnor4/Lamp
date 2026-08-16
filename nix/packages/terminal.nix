{ lib, rustPlatform }:

rustPlatform.buildRustPackage rec {
  pname = "lamp-term";
  version = "0.1.0";
  src = ./../../lamp-term;
  cargoLock.lockFile = ./../../lamp-term/Cargo.lock;

  meta = with lib; {
    description = "Lamp terminal — Genie-chat with /command routing";
    homepage = "https://github.com/falconnor4/Lamp";
    license = licenses.mit;
  };
}