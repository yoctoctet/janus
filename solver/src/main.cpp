#include <iostream>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <janus/janus.hpp>

int main_run(int argc, char *argv[]) { return 0; }
int main_bench(int argc, char *argv[]) { return 0; }
int main_validate(int argc, char *argv[]) { return 0; }

int main(int argc, char *argv[])
{
    CLI::App app;

    CLI::App *app_run = app.add_subcommand("run", "Run a simulation");
    CLI::App *app_bench = app.add_subcommand("bench", "Benchmark the program");
    CLI::App *app_validate = app.add_subcommand("validate", "Validate the methods against a set of standards");

    CLI11_PARSE(app, argc, argv);

    return 0;
}