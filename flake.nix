{
  description = "Data processing for my research project (PhdTrack-Masterarbeit)";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-unstable;
    research-base.url = "github:0nyr/research-base";
  };

  outputs = { self, nixpkgs, research-base, ... }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pythonPackages = pkgs.python311Packages;
    researchPackage = research-base.defaultPackage.${system};
  in {
    # package definition
    defaultPackage.${system} = pythonPackages.buildPythonPackage rec {
      pname = "data-processing-masterarbeit";
      version = "0.1.0";
      src = ./.;

      nativeBuildInputs = with pkgs; [
        pythonPackages.setuptools
        pythonPackages.pytest
      ];

      propagatedBuildInputs = with pythonPackages; [
        python-dotenv
        pandas
        requests
        datetime
        graphviz
        pygraphviz
        networkx
        researchPackage
      ];

    };

    # shell definition
    devShells.${system} = pkgs.mkShell {
      nativeBuildInputs = with pythonPackages; [
        python
        researchPackage
      ];
    };
  };
}