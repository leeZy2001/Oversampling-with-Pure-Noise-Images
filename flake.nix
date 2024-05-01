{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/release-23.11";
  };

  outputs = {self, nixpkgs}: let
    systems = [
      "x86_64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (system: let
        pkgs = import nixpkgs {inherit system;};
      in with pkgs; {
        default = mkShell {
          name = "Python";

          buildInputs = [
            (python3.withPackages (ps: with ps; [
              python-lsp-server
              keras
              tensorflow
              numpy
            ]))
          ];
        };
      }
    );
  };
}
