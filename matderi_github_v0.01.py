# 맛대리 v0.0.1
# suengjae hong

#pip install tabulate

from dotenv import load_dotenv

import os, json, time, requests
from datetime import datetime
import pandas as pd
import numpy as np # numpy==1.26.4
import altair as alt
# import plotly.express as px # a terse and high-level API for creating figures
import streamlit as st

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Langchain only
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

### 함수 Setup
# word cloud dataframe
def transform_to_value_count(df, column_name):
    """
    Transforms a DataFrame with a column of comma-separated values 
    into a value-count DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with one column containing comma-separated values.
        column_name (str): Name of the column to process.

    Returns:
        pd.DataFrame: A DataFrame with unique values and their counts.
    """
    # Step 1: Ensure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Step 2: Split and explode the column
    exploded_series = df[column_name].str.split(',').explode().str.strip()
    
    # Step 3: Count the occurrences
    value_counts = exploded_series.value_counts().reset_index()
    value_counts.columns = [column_name, 'count']
    
    return value_counts

### SETTING ###
## API ###
load_dotenv()
default_api_key = os.environ.get('OPENAI_API_KEY')

## Dataframe ###
FILE_PATH = 'https://raw.githubusercontent.com/suengj/chatbot_matderi/refs/heads/main/yeoeuido_matzip.csv'
df = pd.read_csv(f'{FILE_PATH}')

## FONT ###
FONT_PATH = 'https://raw.githubusercontent.com/suengj/chatbot_matderi/refs/heads/main/NanumGothicLight.ttf'
FONT_LOCAL_PATH = "NanumGothicLight.ttf"

def ensure_font_downloaded(font_url, local_path):
    if not os.path.exists(local_path):
        st.info("Downloading font file. Please wait...")
        response = requests.get(font_url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download font file: {font_url}")

# 웹 대시보드 개발 라이브러리인 스트림릿은 main 함수가 있어야 한다.
def main():

    st.set_page_config(
        page_title = "동여의도 맛대리",
        page_icon = "🤖",
        layout="wide",
        initial_sidebar_state="expanded")

    ## 메시지 ####
    st.title('동여의도 맛인턴_v0.0.1')
    st.header('ChatGPT 기반 동여의도 맛집 DB 검색 봇입니다')

    # Side 창
    st.sidebar.title('필터 입력창')
    st.sidebar.subheader("별도 Open AI API KEY 입력시 다른 모델 사용 가능")

    _usr_key = st.sidebar.text_input('미입력시 기본 제공: GPT 3.5 turbo',
                                     value = "",
                                     type='password') # 미입력시 나의 API로 default 처리 필요

    _usr_key_to_use = default_api_key if _usr_key == "" else _usr_key

    _usr_rate = st.sidebar.number_input('5점 만점 중 평점 입력',
                                        min_value = 0.0,
                                        max_value = 5.0) # 수치 정보 입력

    # _usr_voc = st.sidebar.text_input('원하는 VOC 내용을 입력 하세요') # 현재 기능 updating
    # _usr_select = st.sidebar.multiselect('주요 키워드 선택', ['', '혼밥', '데이트', '무료주차']) # updating

    MODEL_LST = ['gpt-3.5-turbo'] if _usr_key_to_use == default_api_key else ['gpt-4o-2024-08-06','gpt-4o-mini-2024-07-18',
                                                                              'o1-mini-2024-09-12','o1-2024-12-17',
                                                                              'gpt-3.5-turbo']

    _usr_model = st.sidebar.selectbox('선택할 AI 모델', MODEL_LST)

    # _usr_temperature = st.sidebar.slider('창의적 응답이 필요하다면 1에 가깝게 옮기세요',
    #                                      0.0, 1.0, (0.2))
    _usr_temperature = 0.0

    st.sidebar.subheader("version history")

    text_history = """
    **2024-12-20**: 맛인턴 0.0.1. 버젼 작성
    """
    st.sidebar.markdown(text_history)


    # 최종 선택 DB : 필터링 적용
    df['평점'] = pd.to_numeric(df['평점']).fillna(0)

    new_df = df[(df['평점'] >= _usr_rate)]


    ## Agent MODEL & Messages
        
    # Chat AGENT 설정
    LLM_MODEL = ChatOpenAI(
        temperature=_usr_temperature,
        max_tokens = 500,
        model=_usr_model,
        openai_api_key = _usr_key_to_use
    )
    
    # 최초 응답이 없을 경우, 반환하는 내용
    if "messages" not in st.session_state or st.sidebar.button("clear conversation history"):
        st.session_state["messages"] = [{"role":"assistant","content":"무엇을 도와드릴까요"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="원하는 질문을 입력해주세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    if not _usr_key_to_use:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    agent = create_pandas_dataframe_agent(
        LLM_MODEL,    	# 선택된 모델 사용
        new_df,                                	# 데이터프레임
        verbose=False,                      	# 추론과정 출력
        agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


    # separate columns
    col1,col2 = st.columns([2,1])
    
    with col1:
        
        with st.expander('클릭하여 데이터프레임 보기') :
            st.dataframe(new_df)

    with col2:
    
        # word cloud
        # Convert the DataFrame to a dictionary
        _wc_data = transform_to_value_count(new_df, '키워드')
        
        wc_dic = dict(zip(_wc_data['키워드'], _wc_data['count']))
        
        # Generate the Word Cloud
        wc = WordCloud(
            font_path=FONT_LOCAL_PATH,
            width=800, height=400, background_color='white').generate_from_frequencies(wc_dic)

        
        # Display the Word Cloud
        st.write("### 식당 키워드 워드클라우드")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)



if __name__ == '__main__' :
	main()

# end of the code
