{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.terminal;
in {
  options.services.lamp.terminal = {
    enable = lib.mkEnableOption "Lamp terminal";
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ pkgs.lamp.lamp-term ];
  };
}