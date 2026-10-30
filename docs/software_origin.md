# tabfm-distribution 構成ソフトウェアの開発元とライセンス

- 対象: tabfm-distribution（論文「Tabular foundation models for predicting distributions from presence-only species records」の解析コード、https://github.com/hideyuki-doi/tabfm-distribution 、コミット 038a193）
- 作成: 2026-10-28
- 目的: 経済産業省の指針に基づく確認のため、米国由来のプログラムが含まれるか、含まれる場合にオープンソースかを整理する。

## 結論

- 米国が開発元のプログラムは含まれる（依存ソフトの多くと、本体に改変して取り込んだ MaxNet の当てはめ手順）。
- そのうちオープンソースでないのは、**NVIDIA GPU で計算する場合のメーカー製ソフトだけ**である（NVIDIA CUDA：ドライバ、CUDA ランタイム、cuBLAS、cuDNN、NCCL など。GPU 版の PyTorch と一緒にインストールされる）。無料で配布されているが、ソースは非公開。
- 米国以外で、オープンソースでないものが1つある。**TabPFN 2.5 / 3.5 の学習済みの重み**（ドイツ Prior Labs 社）で、誰でも同意の上で無料でダウンロードできるが、非商用・非本番の利用に限るライセンスである。この研究の中心となるモデルなので、TabPFN を使う計算には必須。
- GPU を使わない設定でも主要な計算は動く。PyTorch の CPU 版を使えば NVIDIA のソフトは入らない。ベンチマーク本体（`benchmark_v40_valavi.py`）は GPU がなければ自動で CPU を使い、1種で試すと GPU とほぼ同じ結果（AUC の差 0.0002 以下）で、速さは 5〜15 分の 1 だった。fine-tuning と空間 CV のスクリプトは GPU を前提に書いてあり、CPU で動かすには機器の指定（`device="cuda"`）を変える必要がある。

## 1. パッケージ本体（tabfm-distribution）

本体のコード（Python と R）はこの研究で新しく書いたもので、ライセンスは MIT。ただし、次のプログラムの計算手順を移植・再現した部分を含む。

| 移植・再現した元 | 再現した内容 | 開発元 | ライセンス | オープン |
|---|---|---|---|---|
| maxnet 0.1.4（R） | MaxNet の当てはめ（収束しない場合の予備手順として改変して収録） | 米国（S. Phillips） | MIT | はい |
| scmamp（R） | Friedman aligned-rank 検定と Shaffer の多重比較 | スペイン（バスク大学、B. Calvo・G. Santafé） | GPL-2 | はい |
| prg（R） | AUC-PRG の計算 | 英国（ブリストル大学、M. Kull・P. Flach） | MIT | はい |
| Valavi ら（2022）の解析コード（Presence-Only-SDM） | 参照モデルの設定と手順 | オーストラリア（メルボルン大学、R. Valavi ら） | CC BY-SA 4.0 | はい |
| Valavi ら（2023）の解析コード（SDM-Spatial-Validation） | 空間ブロックへの fold の割り当て | オーストラリア（R. Valavi ら） | 表記なし | ソースは公開、ライセンス表記なし |

次の3つは、配布物には含めず、セットアップ時に元の配布元から自動で取得する（`R/third_party/fetch.sh`）。

| 取得するコード | 用途 | 開発元 | ライセンス | オープン |
|---|---|---|---|---|
| dismo 1.3-14 の gbm.step 一式 | BRT（ブースティング回帰木） | 米国（カリフォルニア大学デービス校、R. Hijmans）。gbm.step の作者はオーストラリア・ニュージーランド（J. Elith・J. Leathwick） | GPL-3 以上 | はい |
| maxent.jar 3.4.4 | MaxEnt | 米国（S. Phillips・M. Dudík・R. Schapire、アメリカ自然史博物館で公開） | MIT | はい |
| myspatial の make_quadratic | Lasso の二次項 | オーストラリア（R. Valavi） | 表記なし | ソースは公開、ライセンス表記なし |

## 2. 動かすのに必要なソフト（利用者が別途インストール）

| ソフト | 用途 | 開発元 | ライセンス | オープン |
|---|---|---|---|---|
| Python | 必須 | 米国（Python Software Foundation） | PSF | はい |
| NumPy、SciPy、pandas | 必須 | 米国中心の開発コミュニティ | BSD | はい |
| scikit-learn、joblib | 必須 | フランス（Inria 発）・国際コミュニティ | BSD-3 | はい |
| statsmodels | 必須（統計解析） | 国際コミュニティ | BSD-3 | はい |
| XGBoost | 必須 | 米国（ワシントン大学発、DMLC） | Apache-2.0 | はい |
| LightGBM、elapid、imbalanced-learn | インストール条件に含む（Python 版の参照モデル用。論文の値は R 版） | 米国（Microsoft 社）、米国（C. Anderson）、フランス（G. Lemaître ら） | MIT | はい |
| geopandas、shapely（GEOS）、pyproj（PROJ）、rasterio（GDAL） | 必須（空間 CV、背景点） | 米国・カナダ中心（OSGeo） | BSD / MIT / LGPL | はい |
| matplotlib、adjustText | 必須（図） | 米国発（J. Hunter）、個人開発者 | PSF 系 / MIT | はい |
| PyTorch | 必須（TabPFN、TabICL） | 米国（Meta 社、PyTorch Foundation） | BSD-3 | はい |
| Triton | GPU 版 PyTorch に付属 | 米国（OpenAI 社） | MIT | はい |
| tabpfn（プログラム） | 必須 | ドイツ（Prior Labs 社） | Apache-2.0 | はい |
| TabPFN 2.5 / 3.5 の学習済みの重み | 必須（初回に自動でダウンロード） | ドイツ（Prior Labs 社）。配布は Hugging Face（米国）経由 | Prior Labs 独自の非商用ライセンス | **いいえ**（無料・公開、非商用・非本番に限る） |
| tabicl（プログラムと重み） | 必須 | フランス（Inria） | BSD-3 | はい |
| huggingface_hub | 必須（重みの取得） | 米国（Hugging Face 社） | Apache-2.0 | はい |
| その他の Python の依存（pydantic、requests、tqdm など約 50） | 必須 | 各国の個人・コミュニティ | MIT / BSD / Apache-2.0 / MPL-2.0 | はい（すべて確認） |
| NVIDIA CUDA（ドライバ、CUDA ランタイム、cuBLAS、cuDNN、NCCL など） | NVIDIA GPU を使うとき（GPU 版 PyTorch に付属） | 米国（NVIDIA 社） | NVIDIA 独自ライセンス | **いいえ**（無料配布、ソース非公開） |
| R 4.5.2 | 必須（参照モデル） | R Core（国際） | GPL-2 / GPL-3 | はい |
| mgcv、nlme、Matrix | 必須 | 英国（S. Wood）、米国、スイスほか | GPL-2 以上 | はい |
| glmnet | 必須 | 米国（スタンフォード大学） | GPL-2 | はい |
| gbm、randomForest | 必須 | 米国（G. Ridgeway ら、L. Breiman・A. Cutler の原作） | GPL-2 以上 | はい |
| maxnet | 必須 | 米国（S. Phillips） | MIT | はい |
| foreach、iterators、Rcpp、RcppEigen、shape、survival（glmnet の依存） | 必須 | 米国（Microsoft 社ほか）・各国 | Apache-2.0 / GPL / LGPL | はい |
| Java（OpenJDK 8） | 必須（maxent.jar の実行） | 米国（Oracle 社）・コミュニティ | GPL-2（Classpath 例外付き） | はい |

## 3. 開発・公開にだけ使ったもの（配布物にもインストール条件にも入っていない）

Claude Code（米国 Anthropic 社、ソース非公開）をコードの作成と実行の補助に使った（論文に明記）。GitHub と gh（米国、gh は MIT）はコードの公開に、LibreOffice、pandoc、TeX は原稿の作成に使っただけで、解析の実行には必要ない。

## 4. 確認のときのポイント

- オープンソースの部分はソースが一般に公開されており、外為法上の「公知の技術」に当たる可能性が高いと考える。
- 非公開の米国製ソフトは NVIDIA CUDA だけで、GPU を使う場合にだけ必要になる。問題になる場合は PyTorch の CPU 版を使えばよい（ベンチマーク本体はそのまま動く。fine-tuning と空間 CV は機器の指定を変える必要があり、計算時間は大幅に増える）。
- TabPFN の学習済みの重みは米国由来ではないが、オープンソースではない（非商用ライセンス）。ただし公開されており、誰でも同意の上で入手できる。fine-tuning した重みはこのライセンス上の派生物なので、配布していない。
- ライセンス表記のない2件（Valavi らの空間ブロックのコードと make_quadratic）はオーストラリア由来で、ソースは GitHub で公開されている。
- 入力データ（NCEAS の生物分布データ：CC BY 4.0、オーストラリアの研究者が作成し米国の OSF で公開、および Valavi らの空間ブロック）はプログラムではない。
- 開発元の国は、パッケージ情報（作者・所属）から分かる範囲で記載した。正式な確認では各プロジェクトのページで裏を取る。
- 移植・再現した部分の元のライセンス（GPL-2、CC BY-SA 4.0、表記なし）と本体の MIT との関係は、輸出管理とは別に確認しておくとよい。
- 該非判定などの最終的な判断は、大学の安全保障輸出管理の担当部署に確認する。

---

ライセンスの種類は、`requirements.txt` から新しく作った実行環境のパッケージのメタデータ（Python の全 97 パッケージ）、R パッケージの DESCRIPTION ファイル、GitHub と Hugging Face の公開情報（TabPFN 3.5 の重みはライセンス本文）で確認した。CPU のみの動作は、GPU を使わない設定でベンチマーク本体を1種（AWT awt01、TabPFN 3.5 / 2.5、TabICL）について実行して確かめた。
