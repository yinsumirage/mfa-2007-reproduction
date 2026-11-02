"""Embed the controlled comparison and Digits thumbnails into an offline demo."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
S=ROOT/'results/visual_comparison'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    data=json.loads((S/'comparison.json').read_text())
    thumbs=json.loads((S/'thumbnails.json').read_text())
    assert len(data['train_ids'])==len(data['train_labels'])
    assert len(data['test_ids'])==len(data['test_labels'])
    assert not set(data['train_ids'])&set(data['test_ids'])
    assert set(data['train_labels'])==set(data['test_labels'])=={0,1,2}
    assert {m['id'] for m in data['methods']}=={'pca','lda','lpp','mfa'}
    for m in data['methods']:
        assert len(m['train_coordinates'])==len(data['train_ids'])
        assert len(m['test_coordinates'])==len(data['test_ids'])==len(m['predictions'])
        assert all(len(v)==2 and all(math.isfinite(x) for x in v) for v in m['train_coordinates']+m['test_coordinates'])
        correct=sum(a==b for a,b in zip(m['predictions'],data['test_labels']))
        assert correct==m['correct'] and len(data['test_ids'])==m['total']
        assert abs(correct/m['total']-m['accuracy'])<1e-9
    assert data['query']['id'] in data['test_ids']
    for i in data['train_ids']+data['test_ids']:assert thumbs[str(i)].startswith('data:image/png;base64,')
    data['thumbnails']=thumbs
    raw=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html=(HERE/'template.html').read_text().replace('__COMPARISON_DATA__',raw)
    assert '__COMPARISON_DATA__' not in html
    (HERE/'index.html').write_text(html)
    manifest={'built_from':{'comparison.json':digest(S/'comparison.json'),'thumbnails.json':digest(S/'thumbnails.json')},'output_sha256':digest(HERE/'index.html'),'output_bytes':(HERE/'index.html').stat().st_size,'offline':True,'script_dependencies':[],'scope':'fixed Digits [0,1,2], shared train-only PCA98 and Euclidean-orthonormal 2D learned subspaces; separate from round2 results','query_id':data['query']['id'],'train':len(data['train_ids']),'test':len(data['test_ids'])}
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
