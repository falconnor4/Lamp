{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.distributed;
in {
  options.services.lamp.distributed = {
    enable = lib.mkEnableOption "Distributed OS mesh networking";
    peerDiscovery = lib.mkOption {
      type = lib.types.enum [ "mdns" "manual" "dht" ];
      default = "mdns";
      description = "Peer discovery method";
    };
    nodeName = lib.mkOption {
      type = lib.types.str;
      default = config.networking.hostName;
      description = "Name of this node in the mesh";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ pkgs.lamp.lamp-shell ];

    systemd.services.lamp-distributed = {
      description = "Lamp Distributed Mesh";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "juicefs.service" ];
      serviceConfig = {
        ExecStart = "${pkgs.lamp.lamp-shell}/bin/lamp-distributed --node-name ${cfg.nodeName} --discovery ${cfg.peerDiscovery}";
        Restart = "on-failure";
        RestartSec = "10";
      };
    };

    networking.firewall.allowedTCPPorts = [ 4210 4211 4212 ];
    networking.firewall.allowedUDPPorts = [ 5353 ];
  };
}