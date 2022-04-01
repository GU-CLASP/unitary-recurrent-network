{ nixpkgs ? import <nixpkgs> {} }:
let
   nixpkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-21.05.tar.gz";
   overlays = []; 
   config = {
     allowUnfree = true;
     cudaSupport = true;
   };
   myNix = import nixpkgs_source {inherit overlays; inherit config;};
in
with myNix.pkgs;
let hp = haskellPackages.override{
      overrides = self: super: {
        pretty-compact = self.callPackage ./pretty-compact.nix {};
        typedflow = self.callPackage ./typedflow.nix {};};};
    ghc = hp.ghcWithPackages (ps: with ps; ([ typedflow cabal-install QuickCheck optparse-applicative]));

    # py = (pkgs.python36.withPackages (ps: [ps.tensorflow-bin ps.tensorflow-addons]));
in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ glibcLocales
                  ghc 
                ];
  shellHook = ''
 export LANG=en_US.UTF-8
 eval $(egrep ^export ${ghc}/bin/ghc)
'';
}

