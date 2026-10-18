"""One small, explicitly scoped plot from the already-recorded v7 arrays."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retina_platform.plotting import curve_plot,font

def main():
    directory=ROOT/'docs/pro_review_supplement'
    status=json.loads((directory/'review_status.json').read_bytes())
    with np.load(directory/'raw_response/transition_background.npz',allow_pickle=False) as control,\
         np.load(directory/'raw_response/transition_stimulus.npz',allow_pickle=False) as stimulus:
        absolute=stimulus['time'].copy();np.testing.assert_array_equal(absolute,control['time'])
        key='midget_RGC_ON.voltage';values=(stimulus[key][0]-control[key][0])*1e6
        metadata=json.loads(stimulus['manifest_json'].item())
        cell=metadata['metadata']['recorded_cell_ids'][key][0]
    relative=absolute-status['response_test']['origin_seconds']
    result=Image.new('RGB',(980,510),'white');draw=ImageDraw.Draw(result)
    draw.text((20,14),'Transition baseline / original anatomical HC / WIP',font=font(20),fill='#142d43')
    draw.text((20,46),'20 ms static scene versus background; identical full checkpoint after 2.1 s numerical preparation',font=font(13),fill='#34495e')
    scene=Image.open(directory/'controlled_image.png').convert('RGB');scene.thumbnail((240,240))
    result.paste(scene,(20,100));draw.text((20,77),'Fixed input image',font=font(13),fill='#34495e')
    panel=curve_plot(relative,values,(700,340),title='Recorded owner midget ON RGC',unit='stimulus - background [uV]')
    result.paste(panel,(270,80))
    draw.text((20,430),'Display conversion: V to uV only. First recorded owner cell; no amplitude normalization.',font=font(12),fill='#34495e')
    draw.text((20,451),'This is not a controlled ON/OFF, receptive-field, latency or spike-statistics test.',font=font(12),fill='#34495e')
    draw.text((20,472),'New functional HC three-ROI acceptance remains incomplete. All numeric arrays are retained unchanged.',font=font(12),fill='#34495e')
    path=directory/'transition_original_HC_response.png';result.save(path)
    status['figure']={'file':path.name,'source_variable':key,'cell_id':cell,'cell_selection':'first recorded owner ID',
        'display_conversion':'V to uV, multiply by1e6','time_axis':'absolute recorded time minus actual stimulus origin, display only',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'new_functional_HC_result':False}
    (directory/'review_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SCOPED_REAL_TRANSITION_FIGURE_READY',path.relative_to(ROOT),flush=True)
if __name__=='__main__':main()
