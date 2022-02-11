import tensorflow as tf
import numpy as np
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])


print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

#default flag setting

flag = {}

flag['model_name'] = 'efficientdet-d0'
flag['logdir'] = '/tmp/deff/'
flag['runmode'] = 'dry'
flag['trace_filename'] = None

flag['threads'] = 0
flag['bm_runs'] = 10
flag['tensorrt'] = None
flag['delete_logdir'] =True
flag['freeze'] = False
flag['use_xla'] =False
flag['batch_size'] = 1

flag['ckpt_path'] = None
flag['export_ckpt'] = None

flag['hparams'] = ''
flag['input_image'] = None
flag['output_image_dir'] = None

flag['input_video'] = None
flag['output_video'] = None

flag['line_thickness'] = None
flag['max_boxes_to_draw'] = 100
flag['min_score_thresh'] = 0.4
flag['nms_method'] = 'hard'

flag['saved_model_dir'] = '/tmp/saved_model'
flag['tfile_path'] = None

# export parameter

MODEL = 'efficientdet-d0'  #@param
import os
def download(m):
  ckpt_path = os.path.join(os.getcwd(), m)
  return ckpt_path

# Download checkpoint.
ckpt_path = download(MODEL)
print('Use model in {}'.format(ckpt_path))


flag['runmode'] = 'saved_model'
flag['model_name'] = 'efficientdet-d0'
flag['ckpt_path'] = ckpt_path
flag['hparams'] = 'image_size=1920x1280'
flag['saved_model_dir'] = 'savedmodel'

# inference parameter

flag['runmode'] = 'saved_model_infer'
flag['model_name'] = 'efficientdet-d0'
flag['saved_model_dir'] = 'savedmodel'
flag['input_image'] = 'img.png'
flag['output_image_dir'] = 'serve_image_out'
flag['min_score_thresh'] = 0.35
flag['max_boxes_to_draw'] = 200

from p_model_inspect import P_ModelInspector
with mirrored_strategy.scope():
  inspector = P_ModelInspector(
    model_name=flag['model_name'],
    logdir=flag['logdir'],
    tensorrt=flag['tensorrt'],
    use_xla=flag['use_xla'],
    ckpt_path=flag['ckpt_path'],
    export_ckpt=flag['export_ckpt'],
    saved_model_dir=flag['saved_model_dir'],
    tflite_path=flag['tfile_path'],
    batch_size=flag['batch_size'],
    hparams=flag['hparams'],
    score_thresh=flag['min_score_thresh'],
    max_output_size=flag['max_boxes_to_draw'],
    nms_method=flag['nms_method'])
  
  driver = inspector.run_model(
      flag['runmode'],
      input_image=flag['input_image'],
      output_image_dir=flag['output_image_dir'],
      input_video=flag['input_video'],
      output_video=flag['output_video'],
      line_thickness=flag['line_thickness'],
      max_boxes_to_draw=flag['max_boxes_to_draw'],
      min_score_thresh=flag['min_score_thresh'],
      nms_method=flag['nms_method'],
      bm_runs=flag['bm_runs'],
      threads=flag['threads'],
      trace_filename=flag['trace_filename'])

  


from p_model_inspect import P_ModelInspector
from PIL import Image

def parameter_step():
  print('run')
  # Serving time batch size should be fixed.
  batch_size = flag['batch_size'] or 1
  all_files = list(tf.io.gfile.glob(flag['input_image']))
  print('all_files=', all_files)
  num_batches = (len(all_files) + batch_size - 1) // batch_size

  for i in range(num_batches):
    batch_files = all_files[i * batch_size:(i + 1) * batch_size]
    height, width = (1280, 1920) # 직접입력
    images = [Image.open(f) for f in batch_files]
    if len(set([m.size for m in images])) > 1:
      # Resize only if images in the same batch have different sizes.
      images = [m.resize(height, width) for m in images]
    raw_images = [np.array(m) for m in images]
    size_before_pad = len(raw_images)
    if size_before_pad < batch_size:
      padding_size = batch_size - size_before_pad
      raw_images += [np.zeros_like(raw_images[0])] * padding_size

    detections_bs = driver.serve_images(raw_images)
    for j in range(size_before_pad):
      img = driver.visualize(raw_images[j], detections_bs[j], max_boxes_to_draw=200,min_score_thresh=0.35)
      img_id = str(i * batch_size + j)
      output_image_path = os.path.join(flag['output_image_dir'], img_id + '.jpg')
      Image.fromarray(img).save(output_image_path)
      print('writing file to %s' % output_image_path)
  return 0


# model export
@tf.function

def distributed_run_step():
    per_replica = mirrored_strategy.run(parameter_step)
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica,axis=None)
distributed_run_step()

