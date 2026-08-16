{
  description = "Lamp Linux — distributed, AI-native OS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixos-hardware.url = "github:NixOS/nixos-hardware";
    driftwm = {
      url = "github:malbiruk/driftwm";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    bitnet.url = "github:microsoft/BitNet";
    dllm.url = "github:ZHZisZZ/dllm";
  };

  outputs = { self, nixpkgs, nixos-hardware, driftwm, bitnet, dllm, ... }: {
    nixosConfigurations = {
      lamp-desktop = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          nixos-hardware.nixosModules.common-pc
          driftwm.nixosModules.driftwm
          ./nixos/configuration.nix
          ./nixos/modules/driftwm.nix
          ./nixos/modules/genie.nix
          ./nixos/modules/terminal.nix
          ./nixos/modules/fruiger-aero.nix
          ./nixos/modules/distributed.nix
          ./nixos/modules/sync.nix
          ./nixos/modules/backwards-compat.nix
        ];
      };

      lamp-mobile = nixpkgs.lib.nixosSystem {
        system = "aarch64-linux";
        modules = [
          ./nixos/configuration.nix
          ./nixos/modules/driftwm.nix
          ./nixos/modules/genie.nix
          ./nixos/modules/terminal.nix
          ./nixos/modules/fruiger-aero.nix
          ./nixos/modules/distributed.nix
          ./nixos/modules/sync.nix
          ./nixos/modules/backwards-compat.nix
        ];
      };
    };

    packages.x86_64-linux = {
      genie = callPackage ./nix/packages/genie.nix {};
      lamp-term = callPackage ./nix/packages/terminal.nix {};
      lamp-shell = callPackage ./nix/packages/lamp-shell.nix {};
      lin = callPackage ./nix/packages/lin.nix {};
    };
  };
}