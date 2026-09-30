"""Paste this entire file into one Kaggle notebook cell; edit REPO_URL."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

REPO_URL = 'https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
BRANCH = ''             # Empty = repository default branch.
PROJECT_SUBDIR = ''     # Auto-detect; set if repo has several copies.
QUICK = False          # True checks wiring with tiny data; not scientific results.
EPOCHS = 30
TRAIN_SCENES = 4096
VALIDATION_SCENES = 200
TEST_SCENES = 200
SEEDS = [11,22,33]
PATH_COUNT = 3
GEOMETRY_SEED = 2026
PAIR_START = 0
BATCH_SIZE = 64


def main():
    u=urlparse(REPO_URL)
    if (u.scheme!='https' or u.hostname!='github.com' or u.username or u.password or
        u.query or u.fragment or 'YOUR_' in REPO_URL or len(u.path.strip('/').split('/'))!=2):
        raise ValueError('Set REPO_URL to your public GitHub repository URL.')
    working=Path('/kaggle/working')
    if not working.is_dir(): raise RuntimeError('Run this cell in Kaggle.')
    run=Path(tempfile.mkdtemp(prefix='controlled_deformation_',dir=working))
    repo=run/'repo';output=run/'results'
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    clone=['git','clone','--depth','1']
    if BRANCH: clone+=['--branch',BRANCH]
    subprocess.run(clone+['--',REPO_URL,str(repo)],env=env,check=True)
    if PROJECT_SUBDIR:
        project=(repo/PROJECT_SUBDIR).resolve()
        if not project.is_relative_to(repo.resolve()): raise ValueError('Subdirectory must be inside repository')
        matches=[project] if (project/'run_experiments.py').is_file() else []
    else: matches=[f.parent for f in repo.rglob('run_experiments.py') if (f.parent/'models.py').is_file()]
    if len(matches)!=1:
        raise RuntimeError('Upload the EXTRACTED experiment files to GitHub. Set PROJECT_SUBDIR if multiple copies exist.')
    project=matches[0]
    subprocess.run([sys.executable,'-m','pip','install','--quiet','-r',str(project/'requirements.txt')],check=True)
    import torch
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU; enable a GPU for the full run')
    command=[sys.executable,'-u',str(project/'run_experiments.py'),'--output',str(output),
             '--epochs',str(EPOCHS),'--train-scenes',str(TRAIN_SCENES),
             '--validation-scenes',str(VALIDATION_SCENES),'--test-scenes',str(TEST_SCENES),
             '--path-count',str(PATH_COUNT),'--geometry-seed',str(GEOMETRY_SEED),
             '--pair-start',str(PAIR_START),'--batch-size',str(BATCH_SIZE),'--seeds',*map(str,SEEDS)]
    if QUICK: command+=['--quick']
    success=False
    try:
        with (run/'run.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                returncode=process.wait()
            finally:
                if process.poll() is None: process.terminate();process.wait()
            if returncode: raise subprocess.CalledProcessError(returncode,command)
        success=True
    finally:
        output.mkdir(exist_ok=True)
        shutil.copy2(run/'run.log',output/'run.log')
        (output/'run_status.json').write_text(json.dumps({'completed':success}))
        (output/'github_provenance.json').write_text(json.dumps({'repository':REPO_URL,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()},indent=2))
        archive=shutil.make_archive(str(run/'experiment_results'),'zip',output)
        print('\nDownload:',archive)
        from IPython.display import display,FileLink
        display(FileLink(str(Path(archive).relative_to(working))))
    print('\nPaste back this report:\n'+(output/'PASTE_BACK.md').read_text())


if __name__=='__main__':main()
