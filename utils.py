"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def create_rag_chain(db_name):
    """
    引数として渡されたDB内を参照するRAGのChainを作成

    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合の処理
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        # 「data」フォルダ直下の各フォルダ名に対して処理
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            # フォルダ内の各ファイルのデータをリストに追加
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    # AIエージェント機能を使う場合の処理
    else:
        # データベース名に対応した、RAG化対象のデータ群が格納されているフォルダパスを取得
        folder_path = ct.DB_NAMES[db_name]
        # フォルダ内の各ファイルのデータをリストに追加
        add_docs(folder_path, docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings()

    # すでに対象のデータベースが作成済みの場合は読み込み、未作成の場合は新規作成する
    if os.path.isdir(db_name):
        db = Chroma(persist_directory=".db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=".db")

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得

    Args:
        folder_path: フォルダのパス
        docs_all: 各ファイルデータを格納するリスト
    """
    files = os.listdir(folder_path)
    for file in files:
        # ファイルの拡張子を取得
        file_extension = os.path.splitext(file)[1]
        # 想定していたファイル形式の場合のみ読み込む
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue
        docs = loader.load()
        docs_all.extend(docs)


def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 会社に関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_service_doc_chain(param):
    """
    サービスに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # サービスに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値
    
    Returns:
        LLMからの回答
    """
    # 顧客とのやり取りに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]


def delete_old_conversation_log(result):
    """
    古い会話履歴の削除

    Args:
        result: LLMからの回答
    """
    # LLMからの回答テキストのトークン数を取得
    response_tokens = len(st.session_state.enc.encode(result))
    # 過去の会話履歴の合計トークン数に加算
    st.session_state.total_tokens += response_tokens

    # トークン数が上限値を下回るまで、順に古い会話履歴を削除
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        # 最も古い会話履歴を削除
        removed_message = st.session_state.chat_history.pop(1)
        # 最も古い会話履歴のトークン数を取得
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        # 過去の会話履歴の合計トークン数から、最も古い会話履歴のトークン数を引く
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行

    Args:
        chat_message: ユーザーメッセージ
    
    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # AIエージェント機能を利用する場合
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        # LLMによる回答をストリーミング出力するためのオブジェクトを用意
        st_callback = StreamlitCallbackHandler(st.container())
        # Agent Executorの実行（AIエージェント機能を使う場合は、Toolとして設定した関数内で会話履歴への追加処理を実施）
        result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
        response = result["output"]
    # AIエージェントを利用しない場合
    else:
        # RAGのChainを実行
        result = st.session_state.rag_chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        # 会話履歴への追加
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=result["answer"])])
        response = result["answer"]

    # LLMから参照先のデータを基にした回答が行われた場合のみ、フィードバックボタンを表示
    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True
    
    return response


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


def calculate_tool(param):
    """
    数値計算や価格計算を行うツール関数

    Args:
        param: 計算に関する質問やリクエスト

    Returns:
        計算結果や関連情報
    """
    import re
    from datetime import datetime
    
    # 簡単な計算例を処理
    try:
        # 数値と演算子を抽出して安全に計算
        if any(op in param for op in ['+', '-', '*', '/', '×', '÷']):
            # 安全な計算のため、evalは使わずに基本的な計算のみ対応
            calculation_examples = [
                "基本的な四則演算をサポートしています。",
                "価格計算: 商品価格 × 数量 = 合計金額",
                "税込み計算: 税抜き価格 × 1.10 = 税込み価格",
                "割引計算: 元の価格 × (1 - 割引率) = 割引後価格"
            ]
            return "計算に関するご質問ですね。" + "\n".join(calculation_examples)
        else:
            return "具体的な計算内容をお教えください。四則演算、価格計算、税計算、割引計算などに対応できます。"
    except Exception as e:
        return "申し訳ございません。計算処理でエラーが発生しました。もう一度お試しください。"


def datetime_tool(param):
    """
    日時に関する情報を提供するツール関数

    Args:
        param: 日時に関する質問やリクエスト

    Returns:
        日時情報や関連情報
    """
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    
    if "現在" in param or "今" in param:
        return f"現在の日時: {current_time.strftime('%Y年%m月%d日 %H時%M分')}"
    elif "営業時間" in param:
        return "営業時間: 平日 9:00-18:00 (土日祝日は休業)"
    elif "期限" in param or "締切" in param:
        # 一週間後を例として提示
        week_later = current_time + timedelta(days=7)
        return f"一般的なお問い合わせの回答期限: {week_later.strftime('%Y年%m月%d日')}頃"
    else:
        return f"現在の日時: {current_time.strftime('%Y年%m月%d日 %H時%M分')}\n営業時間: 平日 9:00-18:00"


def text_analysis_tool(param):
    """
    テキスト分析を行うツール関数

    Args:
        param: 分析対象のテキストや分析リクエスト

    Returns:
        分析結果
    """
    text_length = len(param)
    word_count = len(param.split())
    
    # 簡単な感情分析（キーワードベース）
    positive_words = ["良い", "素晴らしい", "満足", "嬉しい", "ありがとう", "助かる"]
    negative_words = ["悪い", "不満", "困る", "問題", "エラー", "失敗"]
    
    positive_count = sum(1 for word in positive_words if word in param)
    negative_count = sum(1 for word in negative_words if word in param)
    
    sentiment = "中性"
    if positive_count > negative_count:
        sentiment = "ポジティブ"
    elif negative_count > positive_count:
        sentiment = "ネガティブ"
    
    return f"""テキスト分析結果:
文字数: {text_length}文字
単語数: {word_count}語
感情傾向: {sentiment}
キーワード: {', '.join([word for word in positive_words + negative_words if word in param])}"""


def faq_search_tool(param):
    """
    FAQ検索ツール関数

    Args:
        param: FAQ検索クエリ

    Returns:
        関連するFAQ情報
    """
    # 一般的なFAQの例
    faqs = {
        "料金": "サービス料金に関しては、基本プランは月額1,000円から提供しております。詳細は料金ページをご確認ください。",
        "配送": "商品の配送は通常3-5営業日でお届けいたします。お急ぎの場合は特急便（別途料金）もご利用いただけます。",
        "返品": "商品到着後14日以内であれば返品・交換を承っております。詳細は返品ポリシーをご確認ください。",
        "サポート": "カスタマーサポートは平日9:00-18:00で対応しております。メール・電話・チャットでお問い合わせいただけます。",
        "アカウント": "アカウントの作成・変更は公式サイトのマイページから行っていただけます。"
    }
    
    # キーワードマッチング
    for keyword, answer in faqs.items():
        if keyword in param:
            return f"【FAQ】{answer}"
    
    return "該当するFAQが見つかりませんでした。具体的なご質問内容をお聞かせください。"


def contact_info_tool(param):
    """
    連絡先情報提供ツール関数

    Args:
        param: 連絡先に関する質問

    Returns:
        連絡先情報
    """
    contact_info = """
【株式会社EcoTeeお問い合わせ先】
📞 電話: 03-1234-5678
📧 メール: support@ecotee.co.jp
🏢 住所: 〒100-0001 東京都千代田区千代田1-1-1
🕒 営業時間: 平日 9:00-18:00 (土日祝日休業)
💬 チャットサポート: 公式サイトから24時間利用可能
📄 お問い合わせフォーム: https://ecotee.co.jp/contact
"""
    return contact_info


def order_status_tool(param):
    """
    注文状況確認ツール関数（模擬的な情報を提供）

    Args:
        param: 注文に関する質問

    Returns:
        注文状況情報
    """
    from datetime import datetime, timedelta
    
    # 模擬的な注文状況を生成
    current_time = datetime.now()
    order_date = current_time - timedelta(days=2)
    estimated_delivery = current_time + timedelta(days=2)
    
    return f"""
【注文状況（サンプル）】
注文日: {order_date.strftime('%Y年%m月%d日')}
注文番号: ECO-2024-001234
商品: EcoTeeオリジナルTシャツ
状況: 発送準備中
予定配送日: {estimated_delivery.strftime('%Y年%m月%d日')}

※実際の注文状況を確認するには、注文番号とお客様情報が必要です。
詳細は顧客サポートまでお問い合わせください。
"""


def run_company_doc_chain_safe(param):
    """
    会社に関するデータ参照（安全版）

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答またはフォールバック情報
    """
    if st.session_state.get("company_doc_chain") is not None:
        return run_company_doc_chain(param)
    else:
        return "申し訳ございません。現在、会社情報データベースにアクセスできません。直接お問い合わせください。"


def run_service_doc_chain_safe(param):
    """
    サービスに関するデータ参照（安全版）

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答またはフォールバック情報
    """
    if st.session_state.get("service_doc_chain") is not None:
        return run_service_doc_chain(param)
    else:
        return "申し訳ございません。現在、サービス情報データベースにアクセスできません。直接お問い合わせください。"


def run_customer_doc_chain_safe(param):
    """
    顧客とのやり取りに関するデータ参照（安全版）

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答またはフォールバック情報
    """
    if st.session_state.get("customer_doc_chain") is not None:
        return run_customer_doc_chain(param)
    else:
        return "申し訳ございません。現在、顧客情報データベースにアクセスできません。直接お問い合わせください。"


def product_info_tool(param):
    """
    商品情報提供ツール関数

    Args:
        param: 商品に関する質問

    Returns:
        商品情報
    """
    products = {
        "ecotee": {
            "name": "EcoTeeオリジナルTシャツ",
            "price": "2,980円（税込）",
            "description": "環境に優しいオーガニックコットン100%使用",
            "sizes": ["S", "M", "L", "XL"],
            "colors": ["ホワイト", "ブラック", "ネイビー", "グリーン"]
        },
        "hoodie": {
            "name": "EcoTeeパーカー",
            "price": "5,980円（税込）",
            "description": "リサイクル素材を使用した温かいパーカー",
            "sizes": ["S", "M", "L", "XL", "XXL"],
            "colors": ["グレー", "ブラック", "ネイビー"]
        }
    }
    
    # キーワードマッチング
    if "tシャツ" in param.lower() or "ecotee" in param.lower():
        product = products["ecotee"]
        return f"""
【{product['name']}】
価格: {product['price']}
説明: {product['description']}
サイズ: {', '.join(product['sizes'])}
カラー: {', '.join(product['colors'])}
"""
    elif "パーカー" in param.lower() or "hoodie" in param.lower():
        product = products["hoodie"]
        return f"""
【{product['name']}】
価格: {product['price']}
説明: {product['description']}
サイズ: {', '.join(product['sizes'])}
カラー: {', '.join(product['colors'])}
"""
    else:
        return """
【主な商品ラインナップ】
・EcoTeeオリジナルTシャツ: 2,980円（税込）
・EcoTeeパーカー: 5,980円（税込）

詳細な商品情報をお知りになりたい場合は、商品名をお教えください。
"""


def technical_support_tool(param):
    """
    技術サポートツール関数

    Args:
        param: 技術的な問題に関する質問

    Returns:
        技術サポート情報
    """
    if "サイズ" in param or "選び方" in param:
        return """
【サイズ選びのガイド】
・S: 胸囲88-96cm
・M: 胸囲96-104cm  
・L: 胸囲104-112cm
・XL: 胸囲112-120cm

※詳細なサイズチャートは商品ページをご確認ください。
"""
    elif "洗濯" in param or "手入れ" in param:
        return """
【お手入れ方法】
・水温30度以下で洗濯
・漂白剤は使用不可
・陰干し推奨
・アイロンは中温（150度以下）
・ドライクリーニング可

※オーガニック素材のため、丁寧なお手入れをお願いします。
"""
    elif "エラー" in param or "問題" in param:
        return """
【よくある問題と解決方法】
・注文が完了しない → ブラウザのキャッシュをクリアしてお試しください
・サイズが合わない → 14日以内であれば交換可能です
・商品が届かない → 配送状況をお調べいたします

詳細は技術サポート（03-1234-5678）までお問い合わせください。
"""
    else:
        return "技術的なご質問やお困りごとがございましたら、具体的な内容をお教えください。サイズ選び、お手入れ方法、注文に関する問題などにお答えできます。"


def promotion_tool(param):
    """
    プロモーション情報提供ツール関数

    Args:
        param: キャンペーンや割引に関する質問

    Returns:
        プロモーション情報
    """
    from datetime import datetime, timedelta
    
    current_date = datetime.now()
    end_date = current_date + timedelta(days=30)
    
    promotions = f"""
【現在開催中のキャンペーン】

🎉 新規会員登録で10%OFF
期間: 常時開催
対象: 初回購入のお客様
コード: WELCOME10

🌱 エコフレンドリーキャンペーン
期間: {end_date.strftime('%Y年%m月%d日')}まで
内容: 3点以上購入で送料無料
対象: 全商品

💝 まとめ買い割引
内容: 5点以上で15%OFF、10点以上で20%OFF
対象: 全商品（セール品除く）

📧 メルマガ登録特典
内容: 限定クーポンとセール情報をお届け
特典: 登録後すぐに使える5%OFFクーポン

※キャンペーンの詳細は公式サイトをご確認ください。
"""
    return promotions