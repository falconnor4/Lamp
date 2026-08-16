{ config, pkgs, ... }:

{
  system.stateVersion = "24.11";

  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  networking.hostName = "lamp";
  networking.networkmanager.enable = true;

  time.timeZone = "UTC";

  users.users.lamp = {
    isNormalUser = true;
    extraGroups = [ "wheel" "networkmanager" "video" "input" ];
    shell = pkgs.zsh;
  };

  security.polkit.enable = true;

  services.dbus.enable = true;
  services.udisks2.enable = true;

  fonts.packages = with pkgs; [
    noto-fonts
    noto-fonts-cjk
    noto-fonts-emoji
    source-sans
    (nerdfonts.override { fonts = [ "JetBrainsMono" ]; })
  ];

  environment.systemPackages = with pkgs; [
    git
    vim
    wget
    curl
    zsh
    juicefs
    garage
  ];

  programs.zsh.enable = true;
  programs.dconf.enable = true;
}