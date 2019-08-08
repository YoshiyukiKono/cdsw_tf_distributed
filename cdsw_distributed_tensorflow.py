import cdsw
import os, wait, tempfile, time, json, IPython, subprocess

tf_port = 2323

# Clean up the blank proxy environmental variables,
# which confuse tensorflow.
for thing in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'https_proxy', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY', 'all_proxy', 'ALL_PROXY', 'socks_proxy', 'SOCKS_PROXY']:
  if thing in os.environ and os.environ[thing] == '':
    del os.environ[thing]

def tensorboard(fname):
  url = "http://" + os.environ["CDSW_ENGINE_ID"] + ".consoles." + os.environ["CDSW_DOMAIN"]
  tb = "/home/cdsw/.local/bin/tensorboard"
  FNULL = open(os.devnull, 'w')
  proc = subprocess.Popen([tb, "--logdir=%s" % fname, "--port=%s" % os.environ["CDSW_PUBLIC_PORT"]], stdout=FNULL, stderr=FNULL)
  wait.tcp.open(int(os.environ["CDSW_PUBLIC_PORT"]))
  return url, proc.pid    
    
def tensorflow_worker_code(fname, job_name, worker_script):
  if job_name != "worker" and job_name != "ps":
    raise ValueError("job_name must be 'worker' or 'ps'")
  
  if worker_script is None:
    worker_script_import = ""
  else:
    worker_script_import = "import %s" % worker_script
  
  out = """
import os, time, json, wait

__worker_script_import__

# Clean up the blank proxy environmental variables,
# which confuse tensorflow.
for thing in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'https_proxy', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY', 'all_proxy', 'ALL_PROXY', 'socks_proxy', 'SOCKS_PROXY']:
    if thing in os.environ and os.environ[thing] == '':
        del os.environ[thing]

import tensorflow as tf

# Communicate my IP to master.
open("__fname__/__job_name__/" + os.environ["CDSW_IP_ADDRESS"], "w").close()

# Wait for master to tell me the cluster spec.
while True:  
    if os.path.exists("__fname__/cluster.json"):
        break
    else:
        time.sleep(0.1)

clusterSpec = json.loads(open("__fname__/cluster.json").read())
print("Got cluster spec")
print(clusterSpec)
mySpec = "%s:__tf_port__" % os.environ["CDSW_IP_ADDRESS"]
task_index = clusterSpec["__job_name__"].index(mySpec)

cluster = tf.train.ClusterSpec(clusterSpec)
server = tf.train.Server(cluster, job_name="__job_name__", task_index=task_index)    
    """\
      .replace("__fname__", fname)\
      .replace("__job_name__", job_name)\
      .replace("__worker_script_import__", worker_script_import)\
      .replace("__tf_port__", str(tf_port))

  if job_name == "ps" or worker_script is None:
    out += """
server.start()
server.join()
    """
  else:
    out += """
__worker_script__.run(cluster, server, task_index)
  """.replace("__worker_script__", worker_script)
  return out

# TODO: upstream (https://jira.cloudera.com/browse/DSE-4065)
def await_workers(ids):
  print("Awaiting workers...")
  while True:
    status_dict = dict([(worker['id'], worker['status']) for worker in cdsw.list_workers()])
    # print(status_dict)
    done = True
    for id_ in ids:
      if status_dict[id_] == 'failed':
        raise RuntimeError("Worker %s failed" % id_)
      elif status_dict[id_] == 'timedout':
        raise RuntimeError("Worker %s timed out" % id_)
      elif status_dict[id_] == 'stopped':
        raise RuntimeError("Worker %s was stopped" % id_)
      elif status_dict[id_] != 'succeeded':
        # print("Worker %s has not exited, will recheck..." % id_)
        # This worker has not exited, we need to keep waiting
        done = False
        break
    if not done:
      time.sleep(5)
    else:
      return True
    
def run_cluster(n_workers, n_ps, cpu, memory, nvidia_gpu=0, worker_script=None):
  try:
    os.mkdir("/home/cdsw/.tmp", mode=755)
  except:
    pass
  fname = tempfile.mkdtemp(prefix="/home/cdsw/.tmp/clusterspec-")
  os.mkdir(fname + "/worker")
  os.mkdir(fname + "/ps")

  worker_code=tensorflow_worker_code(fname, "worker", worker_script)
  workers = cdsw.launch_workers(n_workers, cpu=cpu, memory=memory, nvidia_gpu=nvidia_gpu, code=worker_code)
  if n_ps > 0:
    ps_code=tensorflow_worker_code(fname, "ps", None)
    parameter_servers = cdsw.launch_workers(n_ps, cpu=cpu, memory=memory, code=ps_code)
  else:
    parameter_servers = []

  while True:
    ips = os.listdir(fname + "/worker")
    if len(ips) != n_workers:
      time.sleep(1)
      continue
    else:
      break

  while True:
    ips = os.listdir(fname + "/ps")
    if len(ips) != n_ps:
      time.sleep(1)
      continue
    else:
      break

  # Atomically write out the cluster spec file
  worker_ips = os.listdir(fname + "/worker")
  ps_ips = os.listdir(fname + "/ps")
  cspec = {
    "worker": [ip + (":%d" % tf_port)for ip in worker_ips],
    "ps": [ip + (":%d" % tf_port) for ip in ps_ips]  
  }
  tmpf = fname + "/cluster.json.tmp"
  f = open(tmpf, 'w')
  f.write(json.dumps(cspec))
  f.flush()
  os.fsync(f.fileno()) 
  f.close()
  os.rename(tmpf, fname + "/cluster.json")
  
  if worker_script is not None:
    # If a script has been provided for the Tensorflow workers,
    # wait for them all to exit.
    await_workers([worker['id'] for worker in  workers])
    cdsw.stop_workers([ps['id'] for ps in parameter_servers])
    return None, None
  else:
    # If no script has been provided, wait for the TensorFlow
    # cluster to come up, then return a handle to the lead worker
    # so the user can create a TensorFlow session.
    
    # Wait for workers to be up
    for ip in worker_ips:
      wait.tcp.open(tf_port, host=ip)

    for ip in ps_ips:
      wait.tcp.open(tf_port, host=ip)

    return cspec, "grpc://%s:%d" % (worker_ips[0], tf_port)
  
