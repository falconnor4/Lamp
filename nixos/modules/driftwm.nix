{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.driftwm;
in {
  options.services.lamp.driftwm = {
    enable = lib.mkEnableOption "DriftWM infinite tiling compositor";
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ pkgs.driftwm ];

    services.displayManager.sessionPackages = [ pkgs.driftwm ];

    services.displayManager.defaultSession = "driftwm";

    security.wrappers = {
      driftwm = {
        source = "${pkgs.driftwm}/bin/driftwm";
        capabilities = "cap_sys_admin+ep";
      };
    };
  };
}