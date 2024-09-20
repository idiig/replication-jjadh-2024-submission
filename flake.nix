{
  description = "A nix-shell environment for the replication of the JJADH paper.";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Define cmdstanr package from unstable channel
        cmdstanr = pkgs.rPackages.buildRPackage {
          name = "cmdstanr";
          src = pkgs.fetchurl {
            url = "https://github.com/stan-dev/cmdstanr/archive/refs/tags/v0.8.0.tar.gz";
            sha256 = "06p3z1c08djf6z3lwh973hl7nw8ljj098lrxadayxp33f7nznnfg";
          };
          propagatedBuildInputs = with pkgs.rPackages; [
            checkmate
            jsonlite
            processx
            R6
            vroom
            posterior
            data_table
          ];
        };

        # Define the core R environment with specific packages
        my-r-env = pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
            languageserver
            IRkernel
            png
            reticulate
            knitr
            glossr
            cmdstanr
            rmarkdown
            tidyverse
            fitdistrplus    # For confirm data distribution
            ggpubr          # For easy plotting
            ggrepel         # For easy annotation
            patchwork       # For combining plots
            scales          # For scale adjustments
            ggokabeito
            ggdist
            brms            # Bayesian modeling through Stan
            broom_mixed     # Convert brms model objects to data frames
            emmeans         # Calculate marginal effects in various ways
            tidybayes       # Manipulate Stan objects in a tidy format
          ];
        };

        # Define the Python environment with specific packages
        my-python-env = pkgs.python312.withPackages (p: with p; [
          pip
          rpds-py
          rpy2
          python-lsp-server
          jupyterlab
          jupyterlab-lsp
          hydra-core
        ]);

        # System packages
        system-packages = with pkgs; [
          quarto
          pandoc
          R
          rstudio
          python312
          glibcLocalesUtf8
          git
          nodejs
          javaldx
          gcc
          busybox
        ];

        additionalExtensions = [
          "jupyterlab-quarto"
        ];

      in
      {
        devShells.default = pkgs.mkShell rec { 
          name = "jjadh-chen-2024-replication";

          LOCALE_ARCHIVE = if pkgs.system == "x86_64-linux" then "${pkgs.glibcLocalesUtf8}/lib/locale/locale-archive" else "";
          LANG = "en_US.UTF-8";
          LC_ALL = "en_US.UTF-8";
          LC_TIME = "en_US.UTF-8";
          LC_MONETARY = "en_US.UTF-8";
          LC_PAPER = "en_US.UTF-8";
          LC_MEASUREMENT = "en_US.UTF-8";

          nativeBuildInputs = [ pkgs.bashInteractive ];
          buildInputs = [
            my-python-env 
            my-r-env
            system-packages
          ];

          # Shell hook to start JupyterLab and provide an environment description
          shellHook = ''
     
            # Rstudio
            Rscript -e 'cmdstanr::install_cmdstan()'

            # Path
            TEMPDIR=$(mktemp -d -p /tmp)
            mkdir -p $TEMPDIR
            cp -r ${pkgs.python312Packages.jupyterlab}/share/jupyter/lab/* $TEMPDIR
            chmod -R 755 $TEMPDIR
            echo "$TEMPDIR is the app directory"

            export JUPYTER_CONFIG_DIR=$TEMPDIR
            export JUPYTER_PATH=$TEMPDIR/share/jupyter
            export JUPYTER_RUNTIME_DIR=$TEMPDIR
            export JUPYTER_DATA_DIR=$TEMPDIR

            export PYTHON_PATH=$(which python3)
            export QUARTO_PATH=$(which quarto)
            export QUARTO_PYTHON=$(which python3)

            export R_HOME=$(R RHOME)
            export RPY2_RHOME=$R_HOME
            export R_LIBS_SITE=$(Rscript -e 'cat(paste(.libPaths(), collapse=":"))')

            # Directoty
            git submodule update --remote --recursive
            mkdir -p artifacts/
            mkdir -p artifacts/figures/

            # Extensions
            ${pkgs.lib.concatMapStrings
                 (s: "python3 -m pip install --no-deps --target=$TEMPDIR ${s}; ")
                 (pkgs.lib.unique
                   ((pkgs.lib.concatMap
                       (d: pkgs.lib.attrByPath ["passthru" "jupyterlabExtensions"] [] d)
                       buildInputs) ++ additionalExtensions))  }

            # Build jupyter lab
            jupyter lab build --app-dir=$TEMPDIR

            # Registeration of IRkernal
            mkdir -p $TEMPDIR/share/jupyter/kernels/ir
            cp -r ${pkgs.rPackages.IRkernel}/library/IRkernel/kernelspec/* $TEMPDIR/share/jupyter/kernels/ir

            # Message
            echo "This is a nix-shell environment for the replication of the JJADH paper."

            # Export installed Python packages and their versions to a file
            pip list > python_dependencies.txt

            # Export installed R packages and their versions to a file
            Rscript -e 'installed.packages()[,c("Package","Version")]' > r_dependencies.txt
          '';
        };
      }
    );
}
