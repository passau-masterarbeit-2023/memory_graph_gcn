{
  description = "Memory graph GCN (PhdTrack-Masterarbeit)";

  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-unstable;
    research-base.url = "github:0nyr/research-base";
  };

  outputs = { self, nixpkgs, research-base, ... }: 
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pythonPackages = pkgs.python311Packages;
    researchPackage = research-base.defaultPackage.${system};
  
    impurePythonEnv = pkgs.mkShell rec {
      name = "impurePythonEnv";
      venvDir = "./.venv";
      buildInputs = [
        pythonPackages.python
        pythonPackages.venvShellHook
        pkgs.autoPatchelfHook

        pythonPackages.python-dotenv
        pythonPackages.psutil
        pythonPackages.pandas
        pythonPackages.numpy
        pythonPackages.seaborn
        pythonPackages.matplotlib
        pythonPackages.scikit-learn

        pythonPackages.mypy
        pythonPackages.pandas-stubs
        pythonPackages.types-psutil

        researchPackage

        # deep learning
        pythonPackages.torch
        pythonPackages.datetime
        pythonPackages.graphviz
        pythonPackages.pygraphviz
        pythonPackages.pydot # needed with networkx
        pythonPackages.networkx
        pythonPackages.pillow
      ];

      postVenvCreation = ''
        unset SOURCE_DATE_EPOCH
        pip install -r requirements.txt
        autoPatchelf ./venv
      '';

      postShellHook = ''
        unset SOURCE_DATE_EPOCH
      '';
    };

  in {
      # Expose the environment as a default package
      defaultPackage.x86_64-linux = impurePythonEnv;
    };
}
