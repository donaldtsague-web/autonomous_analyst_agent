"""Tests de la sandbox Python (Étape 2.3 & cas de validation TEST-3)."""

from src.tools.python_sandbox import execute_python_code, run_python_sandbox


def test_analytical_calculation_mean_and_stdev():
    """TEST-3 : calcul de la moyenne et de l'écart-type d'une série chiffrée."""
    code = (
        "import statistics\n"
        "coûts = [120.0, 150.0, 180.0, 210.0, 240.0]\n"
        "print(f'Moyenne: {statistics.mean(coûts):.2f}')\n"
        "print(f'Ecart-type: {statistics.stdev(coûts):.2f}')\n"
    )
    output = run_python_sandbox.invoke({"code": code})

    assert "[sandbox: OK]" in output
    assert "Moyenne: 180.00" in output
    assert "Ecart-type: 47.43" in output


def test_blocks_forbidden_imports():
    """La sandbox doit bloquer les imports système ou réseau."""
    output = run_python_sandbox.invoke({"code": "import os\nprint(os.getcwd())"})

    assert "[sandbox: ERREUR]" in output
    assert "Import non autorisé" in output


def test_enforces_timeout():
    """Une boucle infinie doit être interrompue par le timeout."""
    result = execute_python_code("while True:\n    pass\n", timeout=1)

    assert result["success"] is False
    assert "délai maximal dépassé" in result["stderr"]
