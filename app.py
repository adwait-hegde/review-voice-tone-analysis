"""Run with streamlit run app.py. Only trusted local checkpoints are accepted."""
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from words_voice.data import read_json
from words_voice.inference import infer, transcript_html, COLORS

st.set_page_config(page_title='Words vs. Voice',layout='wide')
st.title('Words vs. Voice')
st.caption('Emotion-aware customer reviews · intended sentiment, sarcasm, and predicted emotion')
view=st.sidebar.radio('View',['Inference','Dataset inspection (ground truth)'])
if view.startswith('Dataset'):
    st.warning('Dataset inspection: these are metadata labels, not model predictions. Likely synthetic speech; emotion labels are not human-verified perceptions.')
    path=Path('artifacts/data/audit.json')
    if path.exists(): st.json(read_json(path))
    else: st.info('Run words-voice audit first.')
    st.stop()

st.info('Primary training data is likely TTS-generated. Human-speech generalization has not been established. Model scores are uncalibrated.')
def clear_result():
    st.session_state.pop('result',None)

paths={
    'text-only':st.sidebar.text_input('Text checkpoint','artifacts/models/tfidf.joblib'),
    'audio-only':st.sidebar.text_input('Audio checkpoint','artifacts/models/mfcc.joblib'),
    'fusion':st.sidebar.text_input('Fusion checkpoint','artifacts/models/fusion.joblib')}
emotion=st.sidebar.text_input('Emotion checkpoint','artifacts/models/emotion.joblib')
prepared=read_json('artifacts/pretrained-config.json') if Path('artifacts/pretrained-config.json').exists() else {}
asr=st.sidebar.text_input('Cached ASR model or local directory',prepared.get('asr_model','tiny.en'))
st.sidebar.caption('No model downloads occur in the demo. Prepare checkpoints and ASR weights using the CLI.')
upload=st.file_uploader('Upload a spoken review',type=['wav'],on_change=clear_result)
text=st.text_area('Optional supplied transcript',help='Plain text has no timestamps. Leave empty for automatic word-timestamped transcription.',on_change=clear_result)
if upload:
    st.audio(upload)
    if st.button('Analyze review',type='primary'):
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            f.write(upload.getvalue()); f.flush()
            with st.spinner('Running available checkpoints…'):
                try: result=infer(f.name,paths,emotion,text or None,asr)
                except (ValueError,OSError,RuntimeError) as exc:
                    st.error(str(exc)); st.stop()
        st.session_state['result']=result
if 'result' in st.session_state:
    result=st.session_state['result']
    for warning in result['warnings']: st.warning(warning)
    comparisons=[]
    for model,p in result['predictions'].items():
        comparisons.append({'Model':model,'Intended sentiment':p.get('sentiment',{}).get('label','Unavailable'),
                            'Sarcasm':p.get('sarcasm',{}).get('label','Unavailable'), 'Category':p.get('combined','Unavailable'),
                            'Smoke checkpoint':p['smoke_only']})
    if comparisons: st.dataframe(pd.DataFrame(comparisons),hide_index=True)
    else: st.info('No review predictions available. Train checkpoints using the README commands.')
    st.subheader('Transcript')
    if result['colored_transcript']:
        st.markdown(transcript_html(result['colored_transcript']),unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(result['colored_transcript']),hide_index=True)
    elif result['transcript']['text']:
        st.write(result['transcript']['text'])
        st.caption('Supplied text is unaligned; word or segment colours would imply unsupported timing.')
    else: st.info('No transcript available. Supply text or cache an ASR model.')
    st.markdown('Legend: ' + ' '.join(f'<span style="background:{color};color:#171717;padding:4px;border-radius:4px">{name}</span>' for name,color in COLORS.items()),unsafe_allow_html=True)
    if result['emotion_regions']:
        st.subheader('Predicted emotion timeline')
        import altair as alt
        frame=pd.DataFrame(result['emotion_regions']).drop(columns=['scores'])
        chart=alt.Chart(frame).mark_bar().encode(x=alt.X('start:Q',title='Seconds'),x2='end:Q',y=alt.Y('emotion:N',title=None),color=alt.Color('emotion:N',scale=alt.Scale(domain=list(COLORS),range=list(COLORS.values()))),tooltip=['start','end','emotion'])
        st.altair_chart(chart,width='stretch'); st.dataframe(frame,hide_index=True)
        st.caption('Postprocessed sliding-window predictions; silence uses an energy speech-activity detector. Natural emotion boundaries have not been validated.')
    st.caption(f"Inference elapsed: {result['latency_seconds']:.2f} seconds")
    st.download_button('Download analysis JSON',__import__('json').dumps(result,indent=2),'review-analysis.json','application/json')
