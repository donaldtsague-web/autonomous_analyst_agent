"""Étape 2.3 — Sandbox d'exécution Python isolée (sous-processus avec timeout)."""

import ast
import subprocess
import sys
import tempfile
from typing import Any

from langchain_core.tools import tool

from src.logger import get_logger, log_step

STEP = "2.3"
logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15

# Modules système/réseau interdits dans la sandbox d'analyse
FORBIDDEN_MODULES = {
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "winreg",
    "multiprocessing",
    " webbrowser",
    "http",
    "urllib",
    "requests",
    "httpx",
    "ftplib",
    "telnetlib",
}

FORBIDDEN_BUILTINS = {"exec", "eval", "compile", "__import__", "input"}


def _validate_ast(code: str) -> str | None:
    """Vérifie statiquement l'arbre syntaxique pour bloquer les imports et appels dangereux."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return f"Erreur de syntaxe Python : {error}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES or root == "os" or root == "sys" or root == "pathlib":
                    return f"Import non autorisé dans la sandbox : '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES or root in {"os", "sys", "pathlib"}:
                return f"Import non autorisé dans la sandbox : '{node.module}'"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                return f"Appel de fonction non autorisé dans la sandbox : '{node.func.id}'"
            if node.func.id == "open":
                return "Accès fichier ('open') non autorisé dans la sandbox."

    return None


def execute_python_code(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Exécute un script Python dans un sous-processus isolé et retourne stdout/stderr/returncode."""
    validation_error = _validate_ast(code)
    if validation_error:
        log_step(logger, STEP, f"Exécution refusée (sécurité) : {validation_error}")
        return {
            "success": False,
            "stdout": "",
            "stderr": validation_error,
            "returncode": -1,
        }

    log_step(logger, STEP, f"Exécution de code Python en sandbox (timeout={timeout}s)")
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            msg = f"Exécution interrompue : délai maximal dépassé ({timeout}s)."
            log_step(logger, STEP, msg)
            return {
                "success": False,
                "stdout": "",
                "stderr": msg,
                "returncode": -2,
            }

    success = completed.returncode == 0
    log_step(
        logger,
        STEP,
        f"Sandbox terminée (code={completed.returncode}, stdout={len(completed.stdout)} car.)",
    )
    return {
        "success": success,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


@tool
def run_python_sandbox(code: str) -> str:
    """Exécute du code Python dans une sandbox isolée pour effectuer des calculs chiffrés,
    statistiques (moyenne, écart-type, agrégations) ou traitements de données.
    Le code doit utiliser `print(...)` pour afficher ses résultats."""
    result = execute_python_code(code)
    if result["success"]:
        output = result["stdout"] or "(Exécution réussie sans sortie stdout. Utilisez print() pour afficher le résultat.)"
        return f"[sandbox: OK]\n{output}"
    return f"[sandbox: ERREUR]\n{result['stderr'] or result['stdout']}"
