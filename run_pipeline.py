import subprocess
import sys
from pathlib import Path
import argparse

ALL_SCRIPTS = [
	Path("src") / "train_phase1.py",
	Path("src") / "train_phase2.py",
	Path("src") / "generate_report.py",
]

def run_script(path):
	print(f"\n--- Ejecutando: {path} ---\n")
	try:
		subprocess.run([sys.executable, str(path)], check=True)
	except subprocess.CalledProcessError as e:
		print(f"El script {path} falló con el código {e.returncode}")
		raise

def main(step: str = 'all'):
	if step == 'report':
		scripts = [Path('src') / 'generate_report.py']
	else:
		scripts = ALL_SCRIPTS

	for s in scripts:
		if not s.exists():
			print(f"Script no encontrado: {s}")
			continue
		run_script(s)
	print("\nPipeline completo ejecutado ✅")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--step', choices=['all','report'], default='all', help='Paso a ejecutar: all o report')
	args = parser.parse_args()
	main(step=args.step)
