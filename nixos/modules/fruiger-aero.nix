{ config, pkgs, lib, ... }:

let
  cfg = config.services.lamp.theme;
in {
  options.services.lamp.theme = {
    enable = lib.mkEnableOption "Fruiger Aero theme";
    accentColor = lib.mkOption {
      type = lib.types.str;
      default = "#00BFFF";
      description = "Primary accent color (deep sky blue)";
    };
    glassOpacity = lib.mkOption {
      type = lib.types.float;
      default = 0.75;
      description = "Glassmorphism opacity level";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.variables = {
      LAMP_THEME_ACCENT = cfg.accentColor;
      LAMP_THEME_GLASS_OPACITY = toString cfg.glassOpacity;
    };

    environment.etc."lamp/theme.toml".text = ''
      [theme]
      name = "fruiger-aero"
      accent = "${cfg.accentColor}"
      glass_opacity = ${toString cfg.glassOpacity}
      backgrounds = [
        "linear-gradient(135deg, #87CEEB, #98FB98, #FFB6C1, #DDA0DD)",
        "linear-gradient(135deg, #00CED1, #7FFFD4, #FFD700, #FF69B4)"
      ]
      window_decorations = "acrylic"
      font = "Source Sans 3"
      monospace_font = "JetBrains Mono"
      shell_greeting = "✦ Genie at your service. Type /help for commands."
    '';
  };
}