{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.compat;
in {
  options.services.lamp.compat = {
    enable = lib.mkEnableOption "Backwards compatibility shims";
    xdgPortal = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable XDG Desktop Portal for flatpak/snap compat";
    };
    waylandCompat = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable XWayland and wayland compatibility layers";
    };
    posixShims = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "POSIX compatibility shims for traditional Linux apps";
    };
  };

  config = lib.mkIf cfg.enable {
    services.xserver.enable = cfg.xdgPortal;
    services.xserver.desktopManager.runXdgAutostart = cfg.xdgPortal;

    services.xserver.xwayland = {
      enable = cfg.waylandCompat;
    };

    environment.systemPackages = lib.optionals cfg.posixShims [
      pkgs.bash
      pkgs.coreutils
      pkgs.util-linux
      pkgs.gnugrep
      pkgs.gnused
      pkgs.findutils
      pkgs.procps
    ];
  };
}