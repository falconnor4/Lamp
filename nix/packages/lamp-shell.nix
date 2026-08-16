{ lib, rustPlatform }:

rustPlatform.buildRustPackage rec {
  pname = "lamp-shell";
  version = "0.1.0";
  src = ./../../shell;
  cargoLock.lockFile = ./../../shell/Cargo.lock;

  meta = with lib; {
    description = "Lamp shell — DriftWM compositor wrapper with Fruiger Aero bar";
    homepage = "https://github.com/falconnor4/Lamp";
    license = licenses.mit;
  };
}