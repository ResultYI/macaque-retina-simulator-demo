"""Existing BC/AC/RGC observers rebound to the integrated model, no new physics."""
from types import SimpleNamespace
import numpy as np
from retina_sim.bc_functional_v1 import BCFunctionalV1
from retina_sim.ac_functional_v1 import ACFunctionalV1
from retina_sim.rgc_functional_v1 import RGCFunctionalV1
from retina_sim.spatial_group_model import BC,RGC
from retina_sim.ac_vs_model import LEGACY_S
from tools.integrated_far_context_v1 import ROOT,read


class Readouts:
    def __init__(self,arm):
        m=arm.inner;self.arm=arm
        bc=object.__new__(BCFunctionalV1);bc.model=m;bc.populations=tuple(BC)+('rod_BC',);bc.fields={}
        def field(name,pop,unit):bc.fields[name]=dict(population=pop,unit=unit,count=m.counts[pop])
        for pop in BC:
            for name,unit in [('input_native_release','dimensionless'),('generator_activation','dimensionless'),('voltage','V'),
                ('unadapted_release','dimensionless'),('adaptation_activity','dimensionless'),('effective_release','dimensionless')]:field(pop+'__'+name,pop,unit)
        for name,unit in [('input_rod_release_native_mean','dimensionless'),('input_conductance_total','S'),('receptor_gate_mean','dimensionless'),
            ('voltage','V'),('membrane_release_proxy','dimensionless'),('vesicle_resource_x','dimensionless'),('effective_output_event_rate','events/s')]:field('rod_BC__'+name,'rod_BC',unit)
        field('AII_interface__voltage','AII_AC','V');field('AII_interface__gap_current','AII_AC','A')
        c=m.contacts['individual_rod_RBC_contacts'];bc.rod_contacts=c
        bc.rod_weight=np.bincount(c['post'],weights=c['coefficient'],minlength=m.counts['rod_BC']);bc.rod_input_coverage=bc.rod_weight>0
        ac=object.__new__(ACFunctionalV1);ac.model=m;ac.bc=bc;ac.channels=m.channels
        ac.ordinary=tuple(c['pool'] for c in m.channels);ac.pools=(*ac.ordinary,LEGACY_S)
        ac.state_keys=tuple(k for k in m.state_keys if k[0] in ac.pools);ac.slices={};at=0
        for key in ac.state_keys:ac.slices[key]=slice(at,at+m.sizes[key]);at+=m.sizes[key]
        ac.fields=read(ROOT/'reports/layerwise_physiology/AC_functional_v1/IDENTITY.json')['fields']
        for contract in ac.fields.values():contract['count']=m.counts[contract['population']]
        rgc=object.__new__(RGCFunctionalV1);rgc.model=m;rgc.populations=tuple(p for p in RGC if m.counts[p]);rgc.fields=rgc._fields()
        self.bc,self.ac,self.rgc=bc,ac,rgc
        self.fields={prefix+name:contract for prefix,reader in [('BC__',bc),('AC__',ac),('RGC__',rgc)] for name,contract in reader.fields.items()}
        self.state_units={(b.population,b.mechanism,b.name):b.unit for b in (*arm.assembly.state_blocks,*arm.assembly.probability_blocks,*arm.assembly.auxiliary_z1_blocks)}

    def observe(self,t,snapshot):
        cone=snapshot['native_outputs']['release'];inner={k:snapshot['states'][k] for k in self.arm.inner.state_keys}
        values={}
        for prefix,reader,output in [('BC__',self.bc,self.bc.observe(t,inner,cone)),
            ('AC__',self.ac,self.ac.observe_coupled(t,inner,cone)),
            ('RGC__',self.rgc,self.rgc.observe_ports(snapshot['inner_outputs']))]:
            values.update({prefix+key:np.array(value,copy=True) for key,value in output.items()})
        for key,value in snapshot['states'].items():
            name='STATE__'+'__'.join(key);values[name]=np.array(value,copy=True)
            self.fields[name]=dict(population=key[0],unit=self.state_units[key],count=len(value),source_state=list(key))
        if set(values)!=set(self.fields) or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Complete integrated record missing/nonfinite')
        return values
