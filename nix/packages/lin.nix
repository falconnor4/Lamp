{ lib, rustPlatform }:

rustPlatform.buildRustPackage rec {
  pname = "lin";
  version = "0.1.0";
  src = ./../../lin;
  cargoLock.lockFile = ./../../lin/Cargo.lock;

  meta = with lib; {
    description = "Lin language Rust bindings for unified CPU/GPU compute";
    homepage = "https://github.com/falconnor4/Lamp";
    license = licenses.mit;
  };
}