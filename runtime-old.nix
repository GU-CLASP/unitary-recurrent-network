{ nixpkgs ? import <nixpkgs> {} }:
let
#   nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs-channels/archive/34943a7d79beabf179e04a2377ea62963547025d.tar.gz; # https://github.com/NixOS/nixpkgs/archive/4cf0b6ba5d5ab5eb20a88449e0612f4dad8e4c29.tar.gz;
   # nixpkgs_source = /local_dir; # for local directory
   # nixpkgs_source = nixpkgs.fetchFromGitHub { # for safety of checking the hash
   #    owner = "jyp";
   #    repo = "nixpkgs";
   #    rev = "8f00a338f03f55c3c59ff04e5e811b447ed72a90";
   #    sha256 = "0id5i1fs6jdkhrk6jl8zmdc95w3k9xljc8dfl5156zw9ff7xmbfw";
   #  };
   # nixpkgs_source = ~/nixpkgs;
    #   nixpkgs_source = nixpkgs.fetchFromGitHub { # for safety of checking the hash
    #   owner = "obsidiansystems";
    #   repo = "nixpkgs";
    #   rev = "python3.tensorflow_2-update-to-2.3.0";
    #   sha256 = "0fplz9fhzg2vqj8mvphgirrawslnqkkkc9k10jm9352nxgbcv6qs";
    # };

   nixpkgs_source = nixpkgs.fetchFromGitHub { # for safety of checking the hash
      owner = "jyp";
      repo = "nixpkgs";
      rev = "1eb0a7a82791c5fe1b5469d8dc2244b9d4743ac6";
      sha256 = "02dzdky475i7qds6gp1q873ln0vnf39dd2b8g1jjxlwmy4nhj46l";
    };
   overlays = [
     ((import /opt/nix/nvidia-current.nix) nixpkgs_source) # fix version of nvidia drivers to Lark's version
     (self: super:  # define our local packages
         {
          python3 = super.python37.override {
            packageOverrides = python-self: python-super: {
              tensorflow-addons = python-self.callPackage /opt/nix/tensorflow-addons-0.10.0.nix { };
              typeguard = python-self.callPackage /opt/nix/typeguard-2.8.nix { };
              astunparse = python-self.callPackage /opt/nix/astunparse-1.6.3.nix { };
              gast = python-self.callPackage /opt/nix/gast-0.3.3.nix { };
              # tensorboard-plugin-wit = python-self.callPackage /opt/nix/tensorboard-plugin-wit-1.7.0.nix { };
           };};})
     #pkgs_source = ~/repo/nixpkgs;
]; 
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
    py = pkgs.python3;
    pyEnv = py.buildEnv.override {
      extraLibs = with py.pkgs;
        [gast
         h5py
         # tensorboard-plugin-wit
         # astunparse
         tensorflow-bin_2
         tensorflow-addons
];};


    # py = (pkgs.python36.withPackages (ps: [ps.tensorflow-bin ps.tensorflow-addons]));
in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ pyEnv];
}

