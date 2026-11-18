"""Structural and semantic regression tests, independent of benchmark assets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gadra.profile import validate
from gadra.protocol import SLOT_BYTES,DMA_BURST_BYTES,SLOT_SECTORS
from gadra.engine import Memory,ReferenceMemory
from gadra.predictor import query_scores,remaining_correction,DemandLUT,witness_layout
from gadra.baseline import from_orin_measurement,reference_metadata
from gadra.study import merge_scene,metrics,read_manifest,records,energy_efficiency

ROOT=Path(__file__).resolve().parents[1]

def configuration():
    return json.loads((ROOT/'configs/scenesplat_scannet.json').read_text())


def observations(n=256):
    score=np.tile(np.array([.8,.2],np.float32),(4,n,1))
    shape=score.shape[:2]
    return dict(scores=score,winner=np.zeros(shape,np.int32),margin=np.full(shape,.6,np.float32),
        target=np.zeros(n,np.int32),regret=np.zeros(shape,np.float32),cosine=np.ones(shape,np.float32),
        squared_error=np.zeros(shape,np.float32))


class PaperContract(unittest.TestCase):
    def test_all_default_configs_match_storage_and_peak(self):
        for path in (ROOT/'configs').glob('*.json'):
            c=json.loads(path.read_text());report=validate(c)
            self.assertEqual(report['total_storage_bytes'],152*1024)
            self.assertEqual(c['feature_sram_bytes']//SLOT_BYTES,64)
            self.assertEqual(c['arrays']*c['array_rows']*c['array_cols']*2*c['clock_hz'],report['peak_fp32_flops'])
        self.assertEqual(DMA_BURST_BYTES,64)
        self.assertEqual(SLOT_BYTES,128)

    def test_65th_feature_evicts_from_64_slots(self):
        for cls in (Memory,ReferenceMemory):
            with self.subTest(memory=cls.__name__):
                m=cls(configuration(),False)
                for i in range(64):m.access(np.array([i*SLOT_SECTORS],np.int64))
                if cls is Memory:
                    m.update_stats();self.assertEqual(len(m.directories[0]),64)
                self.assertEqual(m.capacities[0],64)
                self.assertEqual(m.stats['dram_read_bytes'],64*64)
                for i in range(64):m.access(np.array([i*SLOT_SECTORS],np.int64))
                if cls is Memory:m.update_stats()
                self.assertEqual(m.stats['directory_hits'],64)
                self.assertEqual(m.stats['dram_read_bytes'],64*64)
                m.access(np.array([64*SLOT_SECTORS],np.int64))
                if cls is Memory:m.update_stats()
                self.assertEqual(m.stats['evictions'],1)
                self.assertEqual(m.stats['dram_read_bytes'],65*64)
                m.access(np.array([0],np.int64))
                if cls is Memory:m.update_stats()
                self.assertEqual(m.stats['evictions'],2)
                self.assertEqual(m.stats['dram_read_bytes'],66*64)

    def test_two_64_byte_bursts_fill_one_slot(self):
        m=Memory(configuration(),False)
        m.access(np.arange(SLOT_SECTORS,dtype=np.int64));m.update_stats()
        self.assertEqual(m.stats['dram_read_bytes'],128)
        self.assertEqual(m.stats['directory_misses'],1)
        m.access(np.arange(SLOT_SECTORS,dtype=np.int64));m.update_stats()
        self.assertEqual(m.stats['dram_read_bytes'],128)

    def test_dirty_eviction_uses_protocol_bursts(self):
        m=Memory(configuration(),False)
        for i in range(65):m.access(np.arange(i*SLOT_SECTORS,(i+1)*SLOT_SECTORS,dtype=np.int64),write=True,ready=1)
        m.update_stats()
        self.assertEqual(m.stats['dram_write_bytes'],128)
        self.assertEqual(m.stats['dram_read_bytes'],0)
        m.finish();m.update_stats()
        self.assertEqual(m.stats['dram_write_bytes'],65*128)

    def test_history_and_response_stores_are_bounded(self):
        m=Memory(configuration(),False)
        base=1<<20;m.register_history(base,128*1024)
        for i in range(65):m.access(np.array([(base+i*1024)//32],np.int64),write=True,ready=0)
        self.assertLessEqual(m.stats['history_peak_bytes'],64*1024)
        self.assertGreater(m.stats['history_dram_write_bytes'],0)
        m.response_access(np.arange(65),write=True)
        self.assertEqual(len(m.response_store.pages),64)
        self.assertLessEqual(m.stats['response_peak_bytes'],64*32)

    def test_l1_normalization_and_cumulative_reversals(self):
        score=np.array([[[9.,1.]],[[1.,9.]],[[9.,1.]]],np.float32)
        obs={'scores':score,'winner':score.argmax(2)}
        z,_=query_scores(obs)
        np.testing.assert_allclose(np.abs(z).sum(2),1)
        self.assertAlmostEqual(float(remaining_correction(obs,'cumulative_l1')[0,0]),3.2,places=6)
        self.assertEqual(float(remaining_correction(obs,'final')[0,0]),0)
        self.assertEqual(float(remaining_correction(obs,'cumulative_l1')[-1,0]),0)
        z,_=query_scores({'scores':np.array([[[0.,0.],[-2.,1.]]],np.float32)})
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z[0,1],[-2/3,1/3])

    def test_nonpilot_future_does_not_change_an_early_exit(self):
        c=configuration();obs=observations();context=np.zeros(256,np.int64)
        predictor=DemandLUT([obs],c,[context])
        stop,witnesses,_=predictor.decide(obs,context)
        index=int(np.flatnonzero(~witnesses)[0]);self.assertEqual(stop[index],1)
        changed=copy.deepcopy(obs);changed['scores'][-1,index]=[.01,.99]
        next_stop,_,_=predictor.decide(changed,context)
        self.assertEqual(next_stop[index],stop[index])

    def test_rtx_and_renamed_old_references_are_rejected(self):
        r={'measured':True,'device':'RTX source','samples':[{'gpu_ms':1.}]}
        with self.assertRaises(ValueError):from_orin_measurement(r)
        with self.assertRaises(ValueError):reference_metadata({'status':'orin_reference','Orin_measured':False})
        r['device']='NVIDIA Orin'
        result=from_orin_measurement(r)
        self.assertTrue(result['Orin_measured'])
        self.assertEqual(result['reference_method'],'orin_cuda_events')
        self.assertEqual(result['orin_measured_ms'],1.)

    def test_scene_overlap_counts_each_original_position_once(self):
        parts=[(np.array([0,1]),np.array([[3.,0.],[0.,3.]])),
               (np.array([1,2]),np.array([[0.,3.],[3.,0.]]))]
        pred,c=merge_scene(parts,np.array([2,7,2]),[2,7])
        np.testing.assert_array_equal(pred,[2,7,2])
        self.assertEqual(int(c.sum()),3)
        self.assertEqual(metrics(c),{'miou':100.,'macc':100.})
        with self.assertRaises(ValueError):merge_scene(parts[:1],np.array([2,7,2]),[2,7])

    def test_capacity_sweep_changes_real_directory_size(self):
        for n in (16,32,64,128,256):
            c=configuration();c.update(cam_entries=n,feature_sram_bytes=n*128)
            validate(c,sensitivity=True)
            m=Memory(c,True)
            self.assertEqual(m.capacities[0],n)
            self.assertEqual(len(m.slots[0]),n)

    def test_complete_scene_manifest_rejects_an_omitted_fragment(self):
        splits={d:{r:[f'{r}_{i}' for i in range(n)] for r,n in
            (('calibration',10),('tuning',5),('test',10))} for d in
            ('scannet','scannetpp','matterport3d')}
        fragments=[]
        for dataset,roles in splits.items():
            for scene in roles['test']:
                for fragment in range(2):
                    fragments.append(dict(dataset=dataset,role='test',scene_id=scene,
                        source_split='validation',fragment_index=fragment,fragment_count=2,raw='local'))
        manifest=dict(schema='gadra-study/v1',splits=splits,models={'SceneSplat':dict(fragments=fragments)})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'study.json';path.write_text(json.dumps(manifest))
            loaded=read_manifest(path)
            self.assertEqual(len(records(loaded,'SceneSplat','test')),60)
            loaded['models']['SceneSplat']['fragments'].pop()
            with self.assertRaises(ValueError):records(loaded,'SceneSplat','test')

    def test_full_depth_array_replay_fits_the_paper_profile(self):
        import pickle
        from gadra.graph import Graph
        from gadra.engine import Simulator
        def tensor(identity,shape,role):
            return dict(storage=identity,version=0,address=identity*65536,
                shape=list(shape),stride=[shape[1],1],bytes=int(np.prod(shape))*4,
                element_bytes=4,role=role,contiguous=True,dtype='torch.float32')
        x=tensor(1,(64,16),'feature');w=tensor(2,(16,64),'weight');y=tensor(3,(64,64),'feature')
        trace=dict(model='SceneSplat',dataset='unit',diagnostic_observations=False,
            events=[dict(kind='linear',op='aten.mm.default',stage='unit',inputs=[x,w],outputs=[y])],
            storages=[dict(id=t['storage'],address=t['address'],bytes=t['bytes']) for t in (x,w,y)],
            output_shape=y['shape'])
        maps=dict(indices={},maps={},checkpoints=[dict(name='final',spec=y,mapping=np.arange(64),after_event=0)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unit.json';path.write_text(json.dumps(trace))
            path.with_name('unit_maps.pkl').write_bytes(pickle.dumps(maps))
            for enabled in (False,True):
                graph=Graph(path);c=configuration()
                sim=Simulator(graph,c,coverage=enabled,order=np.arange(64))
                sim.execute(graph.closure([(graph.events[-1]['outputs'][0],np.arange(64))]))
                result=sim.finish()
                self.assertGreater(result['cycles'],0)
                self.assertEqual(result['statistics']['matrix_flops'],2*64*16*64)

    def test_affine_calibration_preserves_the_fitted_mapping(self):
        from gadra.prepare import fit_affine
        source=np.array([[1.,0.,1.],[0.,1.,1.],[1.,1.,1.],[2.,1.,1.]],np.float32)
        weight=np.arange(3*7,dtype=np.float32).reshape(3,7)/8
        fitted=fit_affine(source,source@weight)
        np.testing.assert_allclose(source@fitted,source@weight,rtol=1e-6,atol=1e-6)
        with self.assertRaises(ValueError):fit_affine(source[:0],(source@weight)[:0])

    def test_cdp_and_joint_feedback_termination_replay(self):
        import json,pickle,tempfile
        from pathlib import Path
        import numpy as np
        from gadra.graph import Graph,rows
        from gadra.engine import Simulator
        from gadra.predictor import DemandLUT
        from gadra.diagnostic_pipeline import complete_feedback
        from gadra.readout import affine_recipe

        c=configuration();n=128

        def tensor(identity,shape,role):
         return dict(storage=identity,version=0,address=identity*65536,shape=list(shape),stride=[shape[1],1],bytes=int(np.prod(shape))*4,element_bytes=4,role=role,contiguous=True,dtype='torch.float32')
        x=tensor(1,(n,4),'feature');events=[];storages=[x];cps=[]
        names=[f'enc.enc{i}.block0' for i in range(1,5)]+[f'dec.dec{i}.block0' for i in (2,1,0)]
        for i,name in enumerate(names):
         w=tensor(2*i+2,(4,4),'weight');y=tensor(2*i+3,(n,4),'feature')
         events.append(dict(kind='linear',op='aten.mm.default',stage=name,inputs=[x,w],outputs=[y]));storages.extend([w,y]);x=y
         cps.append(dict(name=name,spec=y,mapping=np.arange(n),after_event=i,stage_end=True,
          readout_inputs=[y],readout_maps=[np.arange(n)],readout_shape=[5,768],semantic_width=768,
          readout_extra_flops_per_row=0,readout_recipe=affine_recipe('SceneSplat',1),
          readout_recipe_verification=dict(max_abs_error=0.,class_agreement=1.)))
        w=tensor(16,(4,768),'weight');y=tensor(17,(n,768),'feature');storages.extend([w,y])
        events.append(dict(kind='linear',op='aten.mm.default',stage='final',inputs=[x,w],outputs=[y]))
        cps.append(dict(name='final',spec=y,mapping=np.arange(n),after_event=7,readout_inputs=[],readout_maps=[],
         readout_shape=[0,0],semantic_width=768,readout_extra_flops_per_row=0,readout_identity=True))
        trace=dict(model='SceneSplat',dataset='unit',events=events,storages=[dict(id=t['storage'],address=t['address'],bytes=t['bytes']) for t in storages],output_shape=y['shape'])
        score=np.tile(np.array([.8,.2],np.float32),(8,n,1));shape=score.shape[:2]
        obs=dict(scores=score,winner=np.zeros(shape,np.int32),margin=np.full(shape,.6,np.float32),target=np.zeros(n,np.int32),regret=np.zeros(shape,np.float32),cosine=np.ones(shape,np.float32),squared_error=np.zeros(shape,np.float32))
        contexts=np.zeros(n,np.int64);predictor=DemandLUT([obs],c,[contexts]);stops,witnesses,q=predictor.decide(obs,contexts)
        with tempfile.TemporaryDirectory() as d:
         p=Path(d)/'unit.json';p.write_text(json.dumps(trace));p.with_name('unit_maps.pkl').write_bytes(pickle.dumps(dict(indices={},maps={},checkpoints=cps)))
         for b in (False,True):
          g=Graph(p);context=dict(cps[0]);context['spec']=dict(cps[0]['spec'],shape=[1,4]);context['mapping']=contexts
          g.maps.update(response_context=context,observation_begin=0,observation_end=7)
          first=g.closure(g.checkpoint_seeds(stops,witnesses));total=g.closure(g.checkpoint_seeds(stops));second=total.minus(first)
          deployed=dict(c,control_sram_bytes=9000)
          sim=Simulator(g,deployed,coverage=b,order=np.arange(n),demand=True)
          sim.predictor_edges=predictor.edges;sim.predictor_response_edges=predictor.response_edges
          sim.predictor_gate_support=None;sim.predictor_nearest_edges=None
          sim.predictor_response_actions=[a<=.02 for a in predictor.response_table];sim.predictor_nearest_actions=[]
          sim.install_predictor(predictor.storage_bytes());sim.initialize_responses(64)
          sim.execute(first,control=(obs,stops,witnesses,'pilot'));complete_feedback(sim,obs,witnesses)
          sim.execute(second,control=(obs,stops,witnesses,'remaining'));r=sim.finish()
          self.assertGreater(r['cycles'],0)
          self.assertEqual(r['statistics'].get('early_output_rows',0),int((stops<7).sum()))
          self.assertGreater(int((stops<7).sum()),0)

    def test_power_requires_rail_interval_and_dram_boundary(self):
        with self.assertRaises(ValueError):energy_efficiency(1.,{'mean_watts':1.},{'mean_watts':1.})
        p=dict(mean_watts=1.,rail='unit-test',interval=dict(start_ns=0,end_ns=1),excludes_offchip_dram=True)
        self.assertEqual(energy_efficiency(1.,p,p),1.)

if __name__=='__main__':unittest.main()
