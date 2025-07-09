let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/archive/fd487183437963a59ba763c0cc4f27e3447dd6dd.tar.gz";
  pkgs = import nixpkgs {
    config = { };
    overlays = [ ];
  };
in
pkgs.mkShellNoCC {
  venvDir = ".venv";
  packages =
    with pkgs;
    [
      python311
      uv
    ]
    ++ (with pkgs.python311Packages; [
      pip
      venvShellHook
    ]);
}
