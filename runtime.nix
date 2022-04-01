{ nixpkgs ? import <nixpkgs> {} }:
let
   nixpkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-21.05.tar.gz";
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

   overlays = [# ((nixpkgs_source: self: pkgs:
#          with pkgs; {
#            linuxPackages = linuxPackages_5_4.extend (self: super: {
#               nvidia_x11 = callPackage (import (nixpkgs_source + "/pkgs/os-specific/linux/nvidia-x11/generic.nix") {
#                 version = "450.66";
#                 sha256_64bit = "03qj42ppzkc9nphdr9zc12968bb8fc9cpcx5f66y29wnrgg3d1yw";
#                 settingsSha256 = "1677g7rcjbcs5fja1s4p0syhhz46g9x2qqzyn3wwwrjsj7rwaz77";
#                 persistencedSha256 = "01kvd3zp056i4n8vazj7gx1xw0h4yjdlpazmspnsmwg24ijb82x4";
#               }) {
#                 libsOnly = true;
#               };
#             });
#           }
# ) nixpkgs_source)
               # fix version of nvidia drivers];
     ((import /opt/nix/nvidia-current.nix) nixpkgs_source)
      (self: super:  # define our local packages
         {
          # python3 = super.python37.override {
          #   packageOverrides = python-self: python-super: {
          #     tensorflow-addons = python-self.callPackage /opt/nix/tensorflow-addons-0.10.0.nix { };
          #     typeguard = python-self.callPackage /opt/nix/typeguard-2.8.nix { };
          #     astunparse = python-self.callPackage /opt/nix/astunparse-1.6.3.nix { };
          #     gast = python-self.callPackage /opt/nix/gast-0.3.3.nix { };
          #     # tensorboard-plugin-wit = python-self.callPackage /opt/nix/tensorboard-plugin-wit-1.7.0.nix { };
          # };
          # };
           })
     #pkgs_source = ~/repo/nixpkgs;
]; 
   config = {
     allowUnfree = true;
     cudaSupport = true;
   };
   myNix = import nixpkgs_source {inherit overlays; inherit config;};
in
with myNix.pkgs;
let py = pkgs.python3;
    pyEnv = py.buildEnv.override {
      extraLibs = with py.pkgs;
        [
        # gast
         # h5py
         # tensorboard-plugin-wit
         # astunparse
         # tensorflow-bin_2
         tensorflow
         # tensorflow-addons
];};


    # py = (pkgs.python36.withPackages (ps: [ps.tensorflow-bin ps.tensorflow-addons]));
in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ pyEnv];
}

