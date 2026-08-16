{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.sync;
in {
  options.services.lamp.sync = {
    enable = lib.mkEnableOption "JuiceFS + Garage file sync";
    garageEndpoint = lib.mkOption {
      type = lib.types.str;
      default = "http://127.0.0.1:3900";
      description = "Garage S3-compatible endpoint";
    };
    juicefsMount = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/lamp/sync";
      description = "JuiceFS mount point";
    };
    bucketName = lib.mkOption {
      type = lib.types.str;
      default = "lamp-sync";
      description = "Garage bucket for file sync";
    };
  };

  config = lib.mkIf cfg.enable {
    services.garage = {
      enable = true;
      package = pkgs.garage;
      settings = {
        metadata_dir = "/var/lib/garage/meta";
        data_dir = "/var/lib/garage/data";
        rpc_bind_addr = "[::]:3901";
        rpc_public_addr = cfg.garageEndpoint;
        bootstrap_peers = [];
        [s3_api]
        s3_region = "lamp"
        api_bind_addr = "[::]:3900"
      };
    };

    systemd.services.juicefs-sync = {
      description = "JuiceFS mount for Lamp sync";
      wants = [ "garage.service" ];
      after = [ "garage.service" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStartPre = "${pkgs.juicefs}/bin/juicefs format --storage s3 --bucket ${cfg.garageEndpoint}/${cfg.bucketName} sqlite3:///var/lib/juicefs/meta.db lamp-sync";
        ExecStart = "${pkgs.juicefs}/bin/juicefs mount sqlite3:///var/lib/juicefs/meta.db ${cfg.juicefsMount}";
        ExecStop = "${pkgs.juicefs}/bin/juicefs umount ${cfg.juicefsMount}";
      };
    };
  };
}