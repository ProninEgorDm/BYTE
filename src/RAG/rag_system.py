import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class RealEstateRAG:
    def __init__(
        self,
        chroma_db_path: str = "src/RAG/chroma_db",
        embedding_model: str = "cointegrated/rubert-tiny2",
        collection_name: str = "nedvijimost"
    ):
        self.chroma_db_path = Path(chroma_db_path).resolve()
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        print(f" Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection = self.client.get_collection(name=self.collection_name)
        
    def build_index(self, df: pd.DataFrame, batch_size: int = 32) -> None:
        print(f" Building RAG index for {len(df)} apartments...")
        
        df["text_for_embedding"] = (
            df["title"].fillna("").astype(str) + " " +
            # df["description"].fillna("").astype(str) + " " +
            df["address"].fillna("").astype(str) + " " +
            "Метро: " + df["metro"].fillna("").astype(str) + " " +
            "Время до метро: " + df["metro_time"].fillna("").astype(str) + " " +
            "Комнаты: " + df["rooms"].astype(str) + " " +
            "Площадь: " + df["area"].astype(str) + " м2 " +
            "Цена: " + df["price"].astype(str) + " руб " +
            "Этаж: " + df["floor"].astype(str)
        ).str.strip()
        
        df = df[df["text_for_embedding"].str.len() > 20].reset_index(drop=True)
        
        print(" Generating embeddings...")
        texts = df["text_for_embedding"].tolist()
        embeddings = self.embedding_model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = [f"apt_{row['offer_id']}" for _, row in df.iterrows()]
        metadatas = []
        for _, row in df.iterrows():
            metadatas.append({
                "offer_id": str(row["offer_id"]),
                "price": float(row["price_numeric"]) if pd.notna(row["price_numeric"]) else 0,
                "area": float(row["area"]) if pd.notna(row["area"]) else 0,
                "rooms": str(row["rooms"]),
                "metro": str(row["metro"]),
                "address": str(row["address"]),
                "url": str(row["url"]),
                "title": str(row["title"])[:200]
            })
        
        print(" Adding to ChromaDB...")
        batch_db = 100
        for i in range(0, len(ids), batch_db):
            end = min(i + batch_db, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
                documents=texts[i:end]
            )
        print(f" Index built successfully. {len(ids)} apartments indexed.")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar apartments."""
        if self.collection is None:
            raise ValueError("Collection not loaded. Call build_index() first.")
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        similar_apartments = []
        for i in range(len(results["ids"][0])):
            similar_apartments.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })
        return similar_apartments
    
    def generate_report(
        self,
        target_apartment: Dict,
        similar_apartments: List[Dict],
        use_llm: bool = False,
        llm_backend: str = "template",  # "local", "huggingface", "google", "template"
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_tokenizer = None
    ) -> str:
        meta = target_apartment.get("metadata", target_apartment)
        target_price = meta.get("price", 0)
        target_area = meta.get("area", 0)
        target_price_per_m2 = target_price / target_area if target_area > 0 else 0
        
        comp_prices = [apt.get("metadata", apt).get("price", 0) for apt in similar_apartments]
        comp_areas = [apt.get("metadata", apt).get("area", 0) for apt in similar_apartments]
        valid_comps = [(p, a) for p, a in zip(comp_prices, comp_areas) if p > 0 and a > 0]
        
        avg_price = np.mean([p for p, _ in valid_comps]) if valid_comps else target_price
        avg_ppm = np.mean([p/a for p, a in valid_comps]) if valid_comps else target_price_per_m2
        diff_pct = ((target_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

        competitor_summary = ""
        for i, apt in enumerate(similar_apartments[:3], 1):
            am = apt.get("metadata", apt)
            apt_price = am.get("price", 0)
            apt_area = am.get("area", 1)
            apt_ppm = apt_price / apt_area if apt_area > 0 else 0
            price_diff = ((apt_price - target_price) / target_price * 100) if target_price > 0 else 0
            
            competitor_summary += f"""
    {i}. {am.get('title', 'Без названия')[:50]}
        • Цена: {apt_price:,.0f} ₽ ({price_diff:+.1f}% к целевой)
        • Площадь: {apt_area} м² | Цена/м²: {apt_ppm:,.0f} ₽
        • Метро: {am.get('metro', 'Н/Д')} | Адрес: {am.get('address', 'Н/Д')[:40]}"""

        prompt = f"""Ты — эксперт по недвижимости. Проанализируй квартиру кратко и по делу.

    ЦЕЛЕВАЯ КВАРТИРА:
    • Адрес: {meta.get('address', 'Н/Д')}
    • Цена: {target_price:,.0f} ₽
    • Площадь: {target_area} м²
    • Цена за м²: {target_price_per_m2:,.0f} ₽
    • Метро: {meta.get('metro', 'Н/Д')}
    • Комнат: {meta.get('rooms', 'Н/Д')}

    РЫНОЧНАЯ СТАТИСТИКА (аналоги):
    • Средняя цена: {avg_price:,.0f} ₽
    • Средняя цена за м²: {avg_ppm:,.0f} ₽
    • Отклонение цены: {diff_pct:+.1f}%

    ДЕТАЛИ КОНКУРЕНТОВ (топ-3 аналога):
    {competitor_summary}

    ЗАДАЧА:
    Сравни целевую квартиру с конкурентами. Учти:
    - Разницу в цене за м²
    - Локацию (метро, район)
    - Соотношение цены и площади

    Ответь СТРОГО в формате:
    1. ВЫГОДА: [Да/Нет/Сомнительно] + 1 предложение с цифрами
    2. ПЛЮСЫ: [2-3 пункта через запятую, упомяни конкретные преимущества перед аналогами]
    3. МИНУСЫ: [1-2 пункта или "нет явных", упомяни конкретные недостатки vs аналоги]
    4. ВЕРДИКТ: [✅ Хорошая сделка / ⚠️ Средняя / ❌ Переоценена]
    5. КОММЕНТАРИЙ: [Краткий совет для покупателя на основе сравнения с конкурентами]
    """
        pr = f"""ПРОМТ \n \n ЦЕЛЕВАЯ КВАРТИРА:
                • Адрес: {meta.get('address', 'Н/Д')}
                • Цена: {target_price:,.0f} ₽
                • Площадь: {target_area} м²
                • Цена за м²: {target_price_per_m2:,.0f} ₽
                • Метро: {meta.get('metro', 'Н/Д')}
                • Комнат: {meta.get('rooms', 'Н/Д')}

                РЫНОК (аналоги):
                • Средняя цена: {avg_price:,.0f} ₽
                • Средняя цена за м²: {avg_ppm:,.0f} ₽
                • Отклонение цены: {diff_pct:+.1f}% \n \n"""
        if use_llm:
            try:
                if llm_backend == "local" and MLX_AVAILABLE:
                    return pr+self._generate_local(prompt, local_model_path, local_tokenizer)
                elif llm_backend == "huggingface" and HF_AVAILABLE:
                    return pr+self._generate_huggingface(prompt, api_key, model_name)
                elif llm_backend == "google" and GOOGLE_AVAILABLE:
                    return pr+self._generate_google(prompt, api_key, model_name)
                else:
                    print(f" Backend '{llm_backend}' unavailable or not installed. Falling back to template.")
            except Exception as e:
                print(f" LLM generation failed ({llm_backend}): {e}. Falling back to template.")
                
        return self._generate_template_report(
            meta, target_price, target_area, target_price_per_m2,
            avg_price, avg_ppm, diff_pct, similar_apartments
        )

    def _generate_local(self, prompt, model_path, tokenizer):
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError("Local model path not found")
        if not MLX_AVAILABLE:
            raise ImportError("mlx-lm not installed")
            
        # Lazy load
        if not hasattr(self, '_local_model') or self._local_model is None:
            self._local_model, self._local_tokenizer = load(str(model_path))
            
        from mlx_lm import generate
        response = generate(
            model=self._local_model, tokenizer=self._local_tokenizer, 
            prompt=f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=512, temp=0.15, repetition_penalty=1.1
        )
        return response.split("<|assistant|>")[-1].strip()

    def _generate_huggingface(self, prompt, api_key, model_name):
        if not api_key:
            raise ValueError("HF API Key required")
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not installed")
            
        client = InferenceClient(api_key=api_key)
        model = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Формируем структуру чата для задачи conversational
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.15
        )
        
        if isinstance(response, list):
            return response[0].get('generated_text', '').strip()
        
        # Стандартный формат для новых версий
        return response.choices[0].message.content.strip()

    def _generate_google(self, prompt, api_key, model_name):
        if not api_key:
            raise ValueError("Google API Key required")
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai not installed")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
        
    def _generate_template_report(self, meta, target_price, target_area, target_price_per_m2,
                                  avg_price, avg_ppm, diff_pct, similar_apartments) -> str:
        """Rule-based fallback report."""
        if diff_pct < -12: verdict, conf = "✅ ОТЛИЧНАЯ СДЕЛКА", "Высокая"
        elif diff_pct < -5: verdict, conf = "✅ ВЫГОДНО", "Средняя"
        elif diff_pct < 5: verdict, conf = "⚠️ РЫНОЧНАЯ ЦЕНА", "Высокая"
        elif diff_pct < 15: verdict, conf = "⚠️ ЧУТЬ ВЫШЕ РЫНКА", "Средняя"
        else: verdict, conf = f"❌ ПЕРЕОЦЕНЕНА (+{diff_pct:.1f}%)", "Высокая"
        
        pros, cons = [], []
        if target_price_per_m2 < avg_ppm * 0.95: pros.append("Цена за м² ниже средней")
        elif target_price_per_m2 > avg_ppm * 1.1: cons.append("Цена за м² выше рыночной")
        if meta.get("metro"): pros.append(f"Близость к метро ({meta['metro']})")
        if target_area < 30: cons.append("Малая площадь")
        elif target_area > 60: pros.append("Просторная планировка")
        if not pros: pros.append("Соответствует рыночным параметрам")
        if not cons: cons.append("Явных недостатков не выявлено")
        
        report = f"""
╔════════════════════════════════════════════════════════╗
║ 🏠 АНАЛИЗ СДЕЛКИ — {meta.get('address', 'Адрес')}
╚════════════════════════════════════════════════════════╝

📋 ПАРАМЕТРЫ КВАРТИРЫ
Цена: {target_price:,.0f} ₽ | Площадь: {target_area} м² | Цена/м²: {target_price_per_m2:,.0f} ₽
Метро: {meta.get('metro', 'Н/Д')} | Комнат: {meta.get('rooms', 'Н/Д')}

📊 РЫНОЧНОЕ СРАВНЕНИЕ
Средняя цена аналогов: {avg_price:,.0f} ₽ | Отклонение: {diff_pct:+.1f}%

💡 ОЦЕНКА
{verdict} | Уверенность: {conf}

✅ ПРЕИМУЩЕСТВА: {'; '.join(pros)}
❌ НЕДОСТАТКИ: {'; '.join(cons)}

🔍 ТОП-3 АНАЛОГА:
"""
        for i, apt in enumerate(similar_apartments[:3], 1):
            am = apt.get("metadata", apt)
            ap, aa = am.get("price", 0), am.get("area", 1)
            diff = ((ap - target_price) / target_price * 100) if target_price > 0 else 0
            report += f"   {i}. {am.get('title', 'Без названия')} | 💰 {ap:,.0f} ₽ ({diff:+.1f}%) | 📐 {aa} м² | 📍 {am.get('address', '')[:40]}...\n"
            
        return report + "\n💡 Совет: Сравните лично — фото, ремонт и вид из окна могут оправдать разницу в цене.\n"