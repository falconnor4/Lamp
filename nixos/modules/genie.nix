{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.genie;
in {
  options.services.lamp.genie = {
    enable = lib.mkEnableOption "Genie 1-bit LLM service";
    modelPath = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/genie/models";
      description = "Path to Genie model weights";
    };
    listenAddress = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1:4210";
      description = "IPC address for Lamp terminal communication";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ pkgs.lamp.genie ];

    systemd.services.genie = {
      description = "Genie LLM Daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        ExecStart = "${pkgs.lamp.genie}/bin/genied --listen ${cfg.listenAddress} --model-dir ${cfg.modelPath}";
        Restart = "on-failure";
        RestartSec = "5";
        DynamicUser = true;
        StateDirectory = "genie";
      };
    };
  };
}