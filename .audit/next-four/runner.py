"""Fork-only validation. Never commits, pushes, opens PRs, or uses provider keys."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

SLUG=sys.argv[1]
ROOT=Path.cwd()
INPUT=Path(__file__).resolve().parent
E=Path(os.environ['GITHUB_WORKSPACE'])/'evidence'/SLUG
E.mkdir(parents=True,exist_ok=True)
PINS={
 's2':'a271f7d9c4a185e4d0d4e73101f81980d7273737',
 'ParseBench':'e761ed81ff2e6d3ef3a81032d173a90c9f79c52d',
 'ledger':'7d20326887509f06e4ae1c71eaf517cce20b2f7d',
 'chunkhound':'2df9c5a3b03330dd1a317a61c9b2eb8a054b206c',
}
PATHS={
 's2':['sdk/src/session/append.rs'],
 'ParseBench':['README.md','src/parse_bench/evaluation/metrics/parse/rule_based_metric.py','tests/parse_bench/evaluation/evaluators/test_parse_evaluator_thread_pool.py'],
 'ledger':['internal/api/v2/controllers_schema_insert.go','internal/api/v2/schema_chart_properties.go','internal/api/v2/controllers_schema_properties_test.go'],
 'chunkhound':['pyproject.toml','uv.lock','tests/test_mcp_dependency_contract.py'],
}
PRODUCTION={
 's2':['sdk/src/session/append.rs'],
 'ParseBench':['src/parse_bench/evaluation/metrics/parse/rule_based_metric.py'],
 'ledger':['internal/api/v2/controllers_schema_insert.go'],
 'chunkhound':['pyproject.toml'],
}
meta={'repository_slug':SLUG,'base_commit':PINS[SLUG], 'github_run_id':os.getenv('GITHUB_RUN_ID'),
      'audit_commit':os.getenv('GITHUB_SHA'), 'independent_reviewers':0,
      'commands':[], 'overall':'RUNNING'}

def save():
 (E/'verification.json').write_text(json.dumps(meta,indent=2)+'\n')

def run(name, args, timeout=900, required=True, extra_env=None):
 start=time.monotonic()
 env=os.environ.copy()
 if extra_env: env.update(extra_env)
 for key in list(env):
  if key.endswith('API_KEY') or key in {'GH_TOKEN','GITHUB_TOKEN'}:
   env.pop(key,None)
 with (E/(name+'.log')).open('wb') as log:
  proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try: code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid, signal.SIGTERM)
   try: proc.wait(timeout=10)
   except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 record={'name':name,'argv':args,'exit_code':code,'seconds':round(time.monotonic()-start,3),'log':name+'.log'}
 meta['commands'].append(record);save()
 print(json.dumps(record),flush=True)
 if code and required:
  print((E/(name+'.log')).read_text(errors='replace')[-15000:],flush=True)
  raise RuntimeError(f'{name} failed: {code}')
 return code

def junit(name):
 path=E/(name+'.xml')
 if not path.exists(): raise RuntimeError(f'{name}: no JUnit output; cannot count this as a test reproduction')
 root=ET.parse(path).getroot()
 suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
 result={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
 result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
 meta[name]=result;save();return result

def capture():
 if not (ROOT/'.git').exists(): return
 for rel in PATHS[SLUG]:
  p=ROOT/rel
  if p.exists():
   dest=E/'changed_files'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
   old=subprocess.run(['git','show',f'{PINS[SLUG]}:{rel}'],capture_output=True)
   if old.returncode==0:
    d=E/'base_files'/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(old.stdout)
 subprocess.run(['git','add','-N','--',*PATHS[SLUG]],check=False,capture_output=True)
 diff=subprocess.run(['git','diff','--binary','--',*PATHS[SLUG]],capture_output=True,check=True).stdout
 (E/'candidate.patch').write_bytes(diff)
 meta['patch_sha256']=hashlib.sha256(diff).hexdigest()
 meta['file_hashes']={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in PATHS[SLUG] if (ROOT/rel).exists()}
 (E/'worktree-status.txt').write_text(subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout)
 save()

try:
 actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 if actual != PINS[SLUG]: raise RuntimeError(f'Wrong checkout: {actual}')
 meta['input_patch_sha256']=hashlib.sha256((INPUT/(SLUG+'.patch')).read_bytes()).hexdigest()
 run('apply-check',['git','apply','--check',str(INPUT/(SLUG+'.patch'))])
 run('apply',['git','apply',str(INPUT/(SLUG+'.patch'))])
 originals={rel:subprocess.check_output(['git','show',f'HEAD:{rel}']) for rel in PRODUCTION[SLUG]}
 if SLUG=='s2':
  run('rust-version',['rustc','--version'])
  run('nightly-version',['rustc','+nightly','--version'])
  run('format',['rustfmt','+nightly','--edition','2024','sdk/src/session/append.rs'])
 elif SLUG=='ParseBench':
  run('dependencies',['python','-m','pip','install','-e','.[dev,local]','pytest-socket'],timeout=600)
  files=[p for p in PATHS[SLUG] if p.endswith('.py')]
  run('sort-imports',['python','-m','ruff','check','--select','I','--fix',*files])
  run('format',['python','-m','ruff','format',*files])
  run('versions',['python','-m','pip','freeze'])
 elif SLUG=='ledger':
  run('go-version',['go','version'])
  run('format',['gofmt','-w',*PATHS[SLUG]])
 else:
  run('uv-version',['uv','--version'])
  run('generate-lock',['uv','lock'],timeout=600)
  run('dependencies',['uv','sync','--locked','--group','dev'],timeout=900)
  run('format',['uv','run','--no-sync','ruff','format','tests/test_mcp_dependency_contract.py'])
  run('sort-imports',['uv','run','--no-sync','ruff','check','--select','I','--fix','tests/test_mcp_dependency_contract.py'])
  run('versions',['uv','pip','freeze','--python','.venv/bin/python'])
 candidate={rel:(ROOT/rel).read_bytes() for rel in PRODUCTION[SLUG]}
 try:
  for rel,original in originals.items():
   if SLUG=='s2':
    text=candidate[rel].decode()
    marker='#[cfg(test)]\nmod drain_reconnect_tests {'
    if text.count(marker)!=1: raise RuntimeError('Cannot isolate S2 regression module')
    original += ('\n'+marker+text.split(marker,1)[1]).encode()
   (ROOT/rel).write_bytes(original)
  if SLUG=='s2':
   code=run('baseline',['cargo','test','--locked','-p','s2-sdk','--lib','--all-features','drain_reconnect_tests','--','--nocapture'],timeout=1200,required=False)
   log=(E/'baseline.log').read_text()
   matches=re.findall(r'test result: FAILED\. (\d+) passed; (\d+) failed; (\d+) ignored;',log)
   if code!=101 or not matches or int(matches[-1][1])<3: raise RuntimeError('S2 baseline must fail genuine regression assertions, not compilation')
   meta['baseline']={'passed':int(matches[-1][0]),'failed':int(matches[-1][1]),'ignored':int(matches[-1][2])}
  elif SLUG=='ledger':
   code=run('baseline',['go','test','-json','./internal/api/v2','-run','TestInsertSchemaRejectsUnknownChartProperties|TestInsertSchemaKnownChartPropertiesRemainAccepted|TestStoredLegacyChartStillDecodes','-count=1'],timeout=1200,required=False)
   events=[]
   for line in (E/'baseline.log').read_text(errors='replace').splitlines():
    try: events.append(json.loads(line))
    except ValueError: pass
   fails=[x['Test'] for x in events if x.get('Action')=='fail' and x.get('Test')]
   if code!=1 or not any(x.startswith('TestInsertSchemaRejectsUnknownChartProperties/') for x in fails): raise RuntimeError('Ledger baseline must reach HTTP regression assertions')
   meta['baseline']={'failed_test_events':fails,'passed_test_events':[x['Test'] for x in events if x.get('Action')=='pass' and x.get('Test')]}
  else:
   if SLUG=='ParseBench':
    cmd=['python','-m','pytest','-q','--disable-socket','--allow-unix-socket','tests/parse_bench/evaluation/evaluators/test_parse_evaluator_thread_pool.py']
   else:
    run('baseline-package-metadata',['uv','pip','install','--python','.venv/bin/python','--no-deps','-e','.'],timeout=300)
    cmd=['uv','run','--no-sync','pytest','--noconftest','-o','addopts=','-q','tests/test_mcp_dependency_contract.py']
   code=run('baseline',cmd+['--junitxml='+str(E/'baseline.xml')],timeout=300,required=False)
   counts=junit('baseline')
   if code!=1 or counts['errors'] or counts['failures']<3: raise RuntimeError('Baseline must fail assertions without collection/runtime setup errors')
 finally:
  for rel,data in candidate.items(): (ROOT/rel).write_bytes(data)
 if SLUG=='s2':
  run('candidate',['cargo','test','--locked','-p','s2-sdk','--lib','--all-features'],timeout=900)
  run('lint',['cargo','clippy','--locked','-p','s2-sdk','--all-targets','--all-features','--','-D','warnings','--allow','deprecated'],timeout=900)
  run('format-check',['rustfmt','+nightly','--check','--edition','2024','sdk/src/session/append.rs'])
  log=(E/'candidate.log').read_text();matches=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',log)
  meta['candidate']={'summaries':[{'passed':int(a),'failed':int(b),'ignored':int(c)} for a,b,c in matches]}
 elif SLUG=='ledger':
  run('candidate',['go','test','-race','-json','./internal/api/v2','./internal','-count=1'],timeout=1200)
  run('vet',['go','vet','./internal/api/v2','./internal'],timeout=600)
  run('format-check',['bash','-c','test -z "$(gofmt -l internal/api/v2/controllers_schema_insert.go internal/api/v2/schema_chart_properties.go internal/api/v2/controllers_schema_properties_test.go)"'])
 elif SLUG=='ParseBench':
  run('candidate',['python','-m','pytest','-q','--disable-socket','--allow-unix-socket','tests/parse_bench/evaluation/metrics/parse','tests/parse_bench/evaluation/evaluators','--junitxml='+str(E/'candidate.xml')],timeout=900)
  counts=junit('candidate')
  if counts['errors'] or counts['failures']: raise RuntimeError('Candidate tests not clean')
  run('lint',['python','-m','ruff','check',*files])
  run('format-check',['python','-m','ruff','format','--check',*files])
 else:
  run('candidate-package-metadata',['uv','pip','install','--python','.venv/bin/python','--no-deps','-e','.'],timeout=300)
  run('candidate',['uv','run','--no-sync','pytest','--noconftest','-o','addopts=','-q','tests/test_mcp_dependency_contract.py','--junitxml='+str(E/'candidate.xml')],timeout=300)
  counts=junit('candidate')
  if counts['errors'] or counts['failures']: raise RuntimeError('Candidate tests not clean')
  run('lock-check',['uv','lock','--check'],timeout=300)
  run('lint',['uv','run','--no-sync','ruff','check','tests/test_mcp_dependency_contract.py'])
  run('native-and-smoke',['make','dev'],timeout=1200)
  run('mcp-smoke',['uv','run','--no-sync','pytest','tests/mcp/','-v','--junitxml='+str(E/'mcp-smoke.xml')],timeout=480,extra_env={'UV_NO_SYNC':'1'})
  junit('mcp-smoke')
  code=run('full-suite',['uv','run','--no-sync','pytest','tests/','-v','--junitxml='+str(E/'full-suite.xml')],timeout=900,required=False,extra_env={'UV_NO_SYNC':'1'})
  if (E/'full-suite.xml').exists(): junit('full-suite')
  meta['full_suite_passed']=code==0
  if code: meta['remaining_gates']=['Code Research execution','Full upstream test gate did not pass; inspect full-suite.log']
  else: meta['remaining_gates']=['Code Research execution','Cross-platform CI']
 run('diff-check',['git','diff','--check'])
 meta['overall']='TARGETED_VALIDATION_PASSED'
except Exception as exc:
 meta['overall']='BLOCKED'
 meta['error']=str(exc)
 (E/'runner-error.txt').write_text(traceback.format_exc())
 print(traceback.format_exc(),flush=True)
finally:
 try: capture()
 except Exception: (E/'capture-error.txt').write_text(traceback.format_exc())
 save()
if meta['overall']=='BLOCKED': sys.exit(1)
