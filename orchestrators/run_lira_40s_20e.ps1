# Proper LiRA evaluation: 40 shadows, 20 epochs, ResNet-18 / CIFAR-10.
# Built to replace the LiRA addendum table in new_report/Results.tex (line 514).
#
# Shadow cache is keyed by ShadowSpec.hash() = sha256(dataset, model, n_shadow,
# sampling_rate, epochs, label_smoothing, batch_size, seed,
# sample_id_scheme_version). All six calls share the same key, so only the
# first invocation pays the multi-hour training cost.

$ErrorActionPreference = "Stop"
$py = "C:\Users\Hlynur\Documents\FLUL\venv\Scripts\python.exe"
$outdir = "results\lira_40s_20e"

$common = @(
    "--db-dir", "contributions",
    "--dataset", "CIFAR10",
    "--model", "resnet18",
    "--target-client", "0",
    "--attacks", "lira",
    "--non-member-source", "train_pool",
    "--n-members", "1000",
    "--n-nonmembers", "1000",
    "--n-shadow", "40",
    "--shadow-epochs", "20",
    "--shadow-cache", "mia_cache/shadow",
    "--seed", "42",
    "--batch-size", "256"
)

# (db-name, member-source-db-name-or-empty, target, rounds, output-suffix)
$runs = @(
    @("CIFAR10",                                       "",        "original",  "1,5,10,15,20", "mia_original_lira40s20e.json"),
    @("CIFAR10_RETRAINED_no0",                         "CIFAR10", "original",  "0,5,10,15,20", "mia_retrain_lira40s20e.json"),
    @("CIFAR10_UNLEARNED_gradient",                    "",        "unlearned", "1,5,10,15,20", "mia_unlearned_gradient_lira40s20e.json"),
    @("CIFAR10_UNLEARNED_influence",                   "",        "unlearned", "1,5,10,15,20", "mia_unlearned_influence_lira40s20e.json"),
    @("CIFAR10_UNLEARNED_hessian",                     "",        "unlearned", "1,5,10,15,20", "mia_unlearned_hessian_lira40s20e.json"),
    @("CIFAR10_UNLEARNED_class_pruning_p05_ft2",       "",        "unlearned", "1,5,10,15,20", "mia_unlearned_class_pruning_p05_ft2_lira40s20e.json")
)

foreach ($r in $runs) {
    $dbname, $memsrc, $target, $rounds, $outname = $r
    $output = Join-Path $outdir $outname
    $args = @(
        "evaluate_mia_per_round.py",
        "--db-name", $dbname,
        "--target",  $target,
        "--rounds",  $rounds,
        "--output",  $output
    )
    if ($memsrc) {
        $args += @("--member-source-db-name", $memsrc)
    }
    $args += $common

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "===== [$stamp] LiRA-40s-20e :: $dbname / $target =====" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED on $dbname (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "===== [$stamp] ALL 6 LiRA-40s-20e runs complete =====" -ForegroundColor Green
