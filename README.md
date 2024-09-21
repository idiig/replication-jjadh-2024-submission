# Reproduction of XXXX for XXXX

This repository is dedicated to the reproduction of the content from the JJADH paper by Chen2024.
The project uses [`nix`](https://github.com/NixOS/nix) to create a temporary environment for the reproduction of the results.

Clone this repository:

```bash
git clone --recursive https://github.com/idiig/jjadh-replication.git
```

## Getting Started

Installation of `nix` follows [https://nix.dev/install-nix](https://nix.dev/install-nix).
After installation, you need to restart the terminal.

This repository uses `nix flakes` You need to enable Flakes by setting an environment variable in your current shell session. Note that this is a temporary change and will only be effective for the duration of the terminal session:

```bash
export NIX_CONFIG="experimental-features = nix-command flakes"
```

After installing Nix, you can enter the development shell provided by this repository:

```bash
nix develop
```

This command will load the environment defined in `flake.nix` and provide you with all necessary dependencies and configurations to reproduce the content from the JJADH paper.

Then you can enter the Jupyterlab for the reproduction:

```bash
jupyter lab main.ipynb
```

### Removing the Nix Environment

The Nix environment is temporary and can be easily removed after you are done with it. To clean up the Nix environment, simply exit the shell by closing the terminal or running:

```bash
exit
```

If you wish to completely remove the Nix installation from your system, you can do so by running:

```bash
sudo rm -rf /nix
```

## Optional: Using `direnv`

[`direnv`](https://direnv.net/) is a shell extension that automatically loads and unloads environment variables based on your working directory.
It can be used to automate the process of loading the Nix environment whenever you enter the project directory.
When you enter the project directory, `direnv` will detect the `.envrc` file. To enable it, run:

```bash
direnv allow
```

This will automatically load the Nix environment specified in `flake.nix` every time you enter the directory.

## Cleaning Up

After completing your work, you can remove the temporary files and directories created during the session:

```bash
rm -rf /path/to/temporary/files # /tmp/tmp.*
```

Make sure to replace `/path/to/temporary/files` (generally `/tmp/tmp.<hash>`) with the actual path to the temporary files generated by your session.
