const { createApp } = Vue;

const AI_TITLE = '运动处方 · 模型分析';

/* 锻炼重点候选项：模型 goals 命中则勾选，未命中的 goals 作为「其他目标」展示 */
const GOAL_OPTIONS = [
  { text: '提高心肺耐力', kw: ['心肺', '耐力', '有氧'] },
  { text: '增加肌肉力量', kw: ['肌肉', '力量', '抗阻'] },
  { text: '提高柔韧性', kw: ['柔韧', '拉伸', '体前屈'] },
  { text: '减少脂肪', kw: ['减脂', '脂肪', '体重', '减重', 'BMI', '肥胖'] },
  { text: '改善平衡与协调', kw: ['平衡', '协调', '稳定'] },
  { text: '提升速度灵敏', kw: ['速度', '灵敏', '敏捷'] },
  { text: '改善体态姿势', kw: ['体态', '姿势', '脊柱', '含胸'] },
  { text: '养成运动习惯', kw: ['习惯', '依从', '坚持', '兴趣'] },
];

/* 体质测试项目：key 对应档案 fitness_test 字段 */
const FITNESS_ITEMS = [
  { key: 'FHLCJ', label: '肺活量', unit: 'ml' },
  { key: 'WSMCJ', label: '50米跑', unit: '秒' },
  { key: 'LDTYCJ', label: '立定跳远', unit: 'cm' },
  { key: 'ZWTQQCJ', label: '坐位体前屈', unit: 'cm' },
  { key: 'WSMWFPCJ', label: '50米×8往返跑', unit: '秒' },
  { key: 'YFZTSCJ', label: '一分钟跳绳', unit: '个' },
  { key: 'BBMCJ', label: '800米跑', unit: '' },
  { key: 'YQMCJ', label: '1000米跑', unit: '' },
  { key: 'YWQZCJ', label: '一分钟仰卧起坐', unit: '个' },
  { key: 'YTXSCJ', label: '引体向上', unit: '个' },
];

const GENERAL_TIPS = [
  '运动前充分热身，运动后做好拉伸。',
  '抗阻训练时避免憋气，保持均匀呼吸。',
  '关注主观感受（RPE），中等强度应微微出汗、能说话但不能唱歌。',
  '出现不适立即停止运动并告知家长或教师。',
];

/* 展示文字清洗：旧留档会把「知识库暂无针对性资料，建议专业人员复核」这类
   面向内部审计的话写进条目文本，按分句整句剔除，页面不再出现这类字样 */
const META_CLAUSE = /知识库|检索|摘录|针对性资料|资料依据|专业人员复核|未经查表/;

function stripMetaClauses(text) {
  return String(text)
    .split(/(，|。|；|、|：|！|？|,|;|:)/)
    .filter((part, i) => i % 2 === 1 || !META_CLAUSE.test(part))
    .join('')
    .replace(/^[，。；、：！？,;:\s]+/, '')
    .replace(/[，。；、：！？,;:\s]+$/, '');
}

function sanitizePrescription(value) {
  if (typeof value === 'string') return stripMetaClauses(value);
  if (Array.isArray(value)) return value.map(sanitizePrescription);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, sanitizePrescription(v)]));
  }
  return value;
}

/* 旧版留档（动作来自早期的 JSON 动作库）按编号前缀归类；新处方的动作类别由后端从知识库带出 */
const PREFIX_CATEGORY = {
  WUP: '热身', AER: '有氧', RES: '抗阻', COR: '核心',
  FLX: '柔韧', SPD: '速度灵敏', CLD: '放松',
};

/* 课表标签配色：素质类别 → 色组（图例按同一分组合并展示）。
   前半是知识库训练手册的素质分类，后半是旧版留档的分类 */
const CATEGORY_STYLE = {
  心肺耐力: 'aer', 肌肉耐力: 'str', 力量: 'str', 速度: 'spd', 灵敏: 'spd', 协调: 'spd',
  平衡: 'bal', 柔韧: 'cool', 综合: 'other', 常规活动: 'warm',
  热身: 'warm', 有氧: 'aer', 抗阻: 'str', 核心: 'str', 速度灵敏: 'spd', 放松: 'cool',
};
/* 旧版留档（知识库接入前生成）的图例 */
const LEGACY_LEGEND_LABELS = [
  { cls: 'warm', label: '热身' },
  { cls: 'aer', label: '有氧' },
  { cls: 'str', label: '力量 / 核心' },
  { cls: 'spd', label: '速度灵敏' },
  { cls: 'cool', label: '拉伸 / 放松' },
];
const LEGEND_LABELS = [
  { cls: 'aer', label: '心肺耐力' },
  { cls: 'str', label: '力量 / 肌肉耐力' },
  { cls: 'spd', label: '速度 / 灵敏 / 协调' },
  { cls: 'bal', label: '平衡' },
  { cls: 'cool', label: '柔韧拉伸' },
  { cls: 'warm', label: '常规活动' },
];

/* 动作详情弹层的分段：知识库动作条目按原书小标题拆出的各段 */
const DETAIL_SECTIONS = [
  { key: 'purpose', label: '练习功能' },
  { key: 'start', label: '起始姿势' },
  { key: 'steps', label: '练习步骤' },
  { key: 'tips', label: '指导提示' },
  { key: 'safety', label: '安全事项' },
];

/* 空骨架：没有数据时报告版式照常渲染，字段留白而非整页隐藏 */
const EMPTY_FITT_ITEM = { value: '', detail: '' };
const EMPTY_RX = {
  student_id: '',
  name: '',
  generated_at: '',
  versions: {},
  goals: [],
  fitt_vp: {
    frequency: { ...EMPTY_FITT_ITEM }, intensity: { ...EMPTY_FITT_ITEM },
    time: { ...EMPTY_FITT_ITEM }, type: { ...EMPTY_FITT_ITEM },
    volume: { ...EMPTY_FITT_ITEM }, progression: { ...EMPTY_FITT_ITEM },
  },
  weekly_plan: [],
  exercise_details: [],
  intensity_control: {
    target_hr_zone: { lower: null, upper: null, formula_note: '' },
    rpe_range: '', rpe_scale: '', talk_test: '', stop_signals: [], evidence_ids: [],
  },
  risk_notes: [],
  forbidden_or_limited: [],
  // 依据链路：国标评分（S 条目）、知识库摘录（E 条目）、生成时的知识库状态
  assessment: [],
  evidence: [],
  knowledge_base: { available: false, fingerprint: '', retrieval_mode: '', stage: '', documents: [], topics: [] },
  disclaimer: '',
};
const clone = (o) => JSON.parse(JSON.stringify(o));

/* 自行录入表单的默认值：预填李沐阳（BJ2025090314）的完整档案，
   不改任何字段直接提交也能走通保存与生成流程 */
const DEFAULT_FORM = {
  name: '李沐阳', gender: '男', age: '14', grade: '初二',
  height: '168.5', weight: '72.3', fhl: '2680',
  resting_hr: '', blood_pressure: '',
  wsm: '8.9', ldty: '185', zwtqq: '4.2', yfzts: '', wsmwfp: '',
  yqm: '4分32秒', bbm: '', ytxs: '3', ywqz: '',
  chronic: '哮喘（轻度间歇性，近一年未急性发作）',
  medications: '沙丁胺醇气雾剂（按需使用）',
  injuries: '右踝韧带扭伤（2025-04，已恢复，剧烈变向时偶有酸胀）',
  symptoms: '',
};

createApp({
  mixins: [AiPanelMixin],

  data() {
    return {
      rx: clone(EMPTY_RX),
      record: null,
      hasData: false,
      students: [],
      cached: {},
      selectedId: '',
      generating: false,
      stage: '',
      genError: '',
      emergency: null,
      exerciseMap: {},
      dayFilter: '全周',
      pop: { show: false, pinned: false, up: false, x: 0, y: 0, pill: null },
      refPop: null,
      showForm: false,
      form: clone(DEFAULT_FORM),
      formSaving: false,
      formError: '',
      generalTips: GENERAL_TIPS,
    };
  },

  computed: {
    isCached() {
      return !!this.cached[this.selectedId];
    },
    /* ---------- 1 基本信息：原始档案（体检 / 体测 / 伤病史） ---------- */

    /* 基本信息直接取原始档案的体检字段 */
    basic() {
      const e = this.rawExam;
      return { gender: e.XB, age: e.NL, grade: e.SCHOOL_GRADE, height: e.SG, weight: e.TZ };
    },
    /* BMI 由档案身高体重现算，保留 1 位小数 */
    bmi() {
      const h = Number(this.rawExam.SG);
      const w = Number(this.rawExam.TZ);
      if (!h || !w) return '';
      return (w / ((h / 100) ** 2)).toFixed(1);
    },

    rawExam() {
      return (this.record && this.record.physical_exam) || {};
    },
    rawFitness() {
      return (this.record && this.record.fitness_test) || {};
    },
    rawHealth() {
      return (this.record && this.record.health_profile) || {};
    },

    /* 体检数据：血压、心率、肺活量、视力、脊柱筛查 */
    examRows() {
      const e = this.rawExam;
      const v = this.rawHealth.vitals || {};
      const pair = (l, r) => (l || r ? `左 ${l || '—'} / 右 ${r || '—'}` : '');
      const jz = (e.JZ || '').trim();
      return [
        { label: '血压', value: v.blood_pressure, unit: 'mmHg' },
        { label: '安静心率', value: v.resting_heart_rate, unit: '次/分' },
        { label: '肺活量', value: e.FHL, unit: 'ml' },
        { label: '远视力', value: pair(e.ZYSL, e.YYSL) },
        { label: '近视力', value: pair(e.ZYJSL, e.YYJSL) },
        { label: '视力评价', value: e.JSLPJ },
        { label: '脊柱筛查', value: jz, wide: true, warn: !!jz && !jz.includes('未见异常') },
      ];
    },

    /* 国标评分（S 条目）按档案字段名索引：后端按知识库里的评分表查表得到 */
    scoreByKey() {
      const map = {};
      (this.rx.assessment || []).forEach((a) => { map[a.key] = a; });
      return map;
    },
    bmiScore() {
      return this.scoreByKey.BMI || null;
    },

    /* 体测成绩：成绩取自档案 fitness_test；得分与等级取自国标评分（没有评分时只显示成绩） */
    fitnessRows() {
      const ft = this.rawFitness;
      const rows = [];
      FITNESS_ITEMS.forEach((it) => {
        const raw = String(ft[it.key] ?? '').trim();
        if (!raw) return;
        rows.push({
          name: it.label,
          value: it.unit && /^[\d.]+$/.test(raw) ? `${raw} ${it.unit}` : raw,
          score: this.scoreByKey[it.key] || null,
        });
      });
      return rows;
    },

    /* 伤病史：兼容结构化条目与纯文本录入 */
    injuryRows() {
      return (this.rawHealth.injury_history || [])
        .map((it) => {
          if (typeof it === 'string') return it;
          const head = `${it.site || ''}${it.type || ''}`;
          const meta = [it.date, it.status].filter(Boolean).join('，');
          return [head || it.note || '', meta && `（${meta}）`, head && it.note ? ` ${it.note}` : '']
            .filter(Boolean).join('').trim();
        })
        .filter(Boolean);
    },

    /* 慢性病：档案 chronic_conditions 字符串列表 */
    chronicRows() {
      return (this.rawHealth.chronic_conditions || [])
        .map((t) => String(t).trim())
        .filter(Boolean);
    },

    goalOptions() {
      const goals = this.rx.goals || [];
      return GOAL_OPTIONS.map((o) => ({
        text: o.text,
        on: goals.some((g) => o.kw.some((k) => g.includes(k))),
      }));
    },
    extraGoals() {
      const goals = this.rx.goals || [];
      return goals.filter((g) => !GOAL_OPTIONS.some((o) => o.kw.some((k) => g.includes(k))));
    },

    hrText() {
      const z = this.rx.intensity_control.target_hr_zone || {};
      if (z.lower && z.upper) return `${z.lower}–${z.upper} 次/分`;
      return this.rx.fitt_vp.intensity.value || '—';
    },
    stopSignals() {
      const s = this.rx.intensity_control.stop_signals || [];
      return s.length ? s.join('、') : '胸闷、胸痛、头晕、呼吸困难、关节锐痛';
    },

    /* 3 运动方案一览表 */
    planRows() {
      const f = this.rx.fitt_vp;
      const ic = this.rx.intensity_control;
      const z = ic.target_hr_zone || {};
      const rpe = `RPE ${ic.rpe_range || '—'}${ic.rpe_scale ? `（${ic.rpe_scale}）` : ''}`;
      const hrLine = z.lower && z.upper ? `靶心率 ${z.lower}–${z.upper} 次/分　${rpe}` : rpe;
      const row = (label, k, cls, item, detail) => ({
        label, k, cls, value: item.value, detail: detail ?? item.detail, refs: item.evidence_ids || [],
      });
      return [
        row('(Type) 运动类型', 'T', 't2', f.type),
        row('(Intensity) 运动强度', 'I', 'i', f.intensity, `${hrLine}　${f.intensity.detail || ''}`.trim()),
        row('(Time) 运动时间', 'T', 't', f.time),
        row('(Frequency) 运动频率', 'F', 'f', f.frequency),
        row('(Volume) 周运动总量', 'V', 'v', f.volume),
        row('(Progression) 运动进阶', 'P', 'p', f.progression, f.progression.detail || ''),
      ];
    },

    /* ---------- 4 一周训练课表 ---------- */

    /* 周计划 → 课表行：每天一行，动作按 热身→主体→放松 的块顺序平铺成标签 */
    schedule() {
      return (this.rx.weekly_plan || []).map((d) => {
        const pills = [];
        (d.blocks || []).forEach((b, bi) => {
          (b.items || []).forEach((it, ii) => {
            const category = this.categoryOf(it);
            pills.push({
              key: `${d.day}-${bi}-${ii}`,
              id: it.exercise_id,
              name: it.name,
              dosage: it.dosage || '',
              notes: it.notes || '',
              phase: b.phase || '',
              category,
              cls: CATEGORY_STYLE[category] || 'other',
              refs: it.evidence_ids || [],
              // 常规活动的名称没有在知识库摘录里找到：后端标记为待核实
              unverified: it.verified === false,
            });
          });
        });
        return {
          day: d.day,
          theme: d.theme || (d.is_training_day ? '训练日' : '休息'),
          rest: !d.is_training_day,
          rationale: d.rationale || '',
          refs: d.evidence_ids || [],
          pills,
          minutes: d.total_minutes ? `约 ${d.total_minutes} 分钟` : '—',
        };
      });
    },
    visibleSchedule() {
      if (this.dayFilter === '全周') return this.schedule;
      return this.schedule.filter((d) => d.day === this.dayFilter);
    },
    dayTabs() {
      return ['全周', ...this.schedule.map((d) => d.day)];
    },
    /* 图例只列出本处方实际用到的类别组 */
    legendCats() {
      const seen = new Set();
      this.schedule.forEach((d) => d.pills.forEach((p) => seen.add(p.cls)));
      const labels = this.hasEvidenceChain ? LEGEND_LABELS : LEGACY_LEGEND_LABELS;
      return labels.filter((l) => seen.has(l.cls));
    },
    /* 概览条：类型 · 强度（靶心率）· 进阶，取自 FITT-VP */
    planBanner() {
      const f = this.rx.fitt_vp;
      const z = this.rx.intensity_control.target_hr_zone || {};
      const hr = z.lower && z.upper ? `靶心率 ${z.lower}–${z.upper} 次/分` : '';
      return [f.type.value, f.intensity.value, hr, f.progression.value]
        .filter(Boolean).join(' · ');
    },
    /* 弹层详情：知识库动作条目的各段文字与出处；旧版留档则是要点与常见错误 */
    popDetail() {
      const d = this.pop.pill ? this.detailOf(this.pop.pill.id) : {};
      return {
        sections: DETAIL_SECTIONS
          .map((sec) => ({ label: sec.label, text: d[sec.key] || '' }))
          .filter((sec) => sec.text),
        equipment: (d.equipment || []).filter((e) => e && e !== '无').join('、'),
        dosage_hint: d.dosage_hint || '',
        source: d.source || '',
        key_points: d.key_points || [],
        common_errors: d.common_errors || [],
      };
    },

    /* ---------- 依据引用与卡片 ---------- */

    evidenceMap() {
      const map = {};
      (this.rx.evidence || []).forEach((e) => { map[e.id] = e; });
      (this.rx.assessment || []).forEach((a) => { map[a.id] = a; });
      return map;
    },
    /* 这份处方有没有依据链路：知识库改造之前生成的留档没有 */
    hasEvidenceChain() {
      return !!(this.rx.knowledge_base && this.rx.knowledge_base.available);
    },
    /* 依据卡片内容：把当前弹出的条目归一成卡片字段（S = 国标评分，E = 知识库摘录） */
    refCard() {
      const e = this.refPop;
      if (!e) return null;
      if (String(e.id).startsWith('S')) {
        return {
          id: e.id,
          fact: true,
          title: `${e.name}：${e.raw}${e.unit ? ` ${e.unit}` : ''}`,
          sub: e.source || '',
          score: e.score,
          level: e.level,
          text: e.basis || '',
        };
      }
      return {
        id: e.id,
        fact: false,
        title: e.doc_title || e.source || '',
        sub: [e.section, e.pages && `第 ${e.pages} 页`].filter(Boolean).join(' · ') || e.source || '',
        kind: e.doc_kind || '',
        topics: e.topics || [],
        citedBy: e.cited_by || [],
        text: e.content || '',
      };
    },

  },

  watch: {
    /* 切换学生：有已生成结果就直接载入，没有就清空版面等待生成。
       录入新档案时 selectedId 由代码设置，由该流程处理，这里跳过 */
    selectedId(id) {
      if (!id || this.generating) return;
      if (this.hasData && this.rx.student_id === id) return;
      this.onStudentChange();
    },
    /* 切换日期标签时收起已钉住的动作详情弹层（全周视图不展示弹层） */
    dayFilter() {
      this.closePop();
    },
  },

  mounted() {
    // 动作详情弹层：点击标签与弹层以外的区域关闭；滚动后 fixed 定位失效，直接收起。
    // 依据卡片（.ev-pop）打开期间的点击与滚动不收起底下的动作弹层，关掉卡片后还能接着看
    document.addEventListener('click', (e) => {
      if (!this.pop.show || !this.pop.pinned) return;
      if (e.target.closest('.move-pop, .ev-pop') || e.target.closest('.sd-items .pill')) return;
      this.closePop();
    });
    // 滚动收起弹层只针对页面滚动；弹层自身内容超高时的内部滚动（scroll 不冒泡但会被
    // window 捕获监听收到）不能收起，否则卡片里一滚就消失
    window.addEventListener('scroll', (e) => {
      if (!this.pop.show) return;
      if (e.target instanceof Element && e.target.closest('.move-pop')) return;
      if (this.refPop) return;
      this.closePop();
    }, true);
    // Esc 关闭依据卡片
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.refPop) this.closeRef();
    });

    // 学生列表与已有结果清单都到位后再定位默认学生，
    // 这样首个学生有已生成结果时能直接自动载入
    const studentsReady = fetch('/api/v1/students')
      .then((r) => r.json())
      .then((list) => { this.students = list; })
      .catch(() => {});

    Promise.all([studentsReady, this.refreshCached()]).then(() => {
      if (this.selectedId) return;
      // 优先认领 URL 携带的学生（如从档案理解报告页返回），无参再退回首个学生
      const wanted = new URLSearchParams(location.search).get('id');
      const hit = wanted && this.students.find((s) => s.student_id === wanted);
      this.selectedId = hit ? hit.student_id
        : (this.students[0] && this.students[0].student_id) || '';
    });
  },

  methods: {
    /* ---------- 切换学生 / 预生成结果 ---------- */

    onStudentChange() {
      this.showForm = false;
      if (this.isCached) this.loadCached();
      else this.resetState();
    },

    refreshCached() {
      return fetch('/api/v1/pregenerated')
        .then((r) => r.json())
        .then((d) => {
          const map = {};
          (d.items || []).forEach((it) => { map[it.student_id] = it; });
          this.cached = map;
        })
        .catch(() => {});
    },

    async loadCached() {
      if (!this.isCached || this.generating) return;
      // 报告版面挂在 v-if="!showForm" 下：先收起录入表单，结果才看得见
      this.showForm = false;
      this.resetState();
      // 原始档案单独拉取：报告头部的基本信息 / 体检 / 体测 / 伤病史直接展示档案原文
      this.fetchRecord(this.selectedId);
      try {
        const res = await fetch(`/api/v1/pregenerated/${this.selectedId}`);
        if (!res.ok) {
          this.genError = `载入失败：${(await res.json()).detail || res.status}`;
          return;
        }
        const data = await res.json();

        if (data.emergency) { this.emergency = data.emergency; }

        if (data.prescription) {
          this.applyResult(data.prescription);
        } else if (!data.emergency) {
          this.genError = '该学生的已有结果不含处方内容，可点「生成处方」生成';
        }
      } catch (err) {
        this.genError = `载入失败：${err}`;
      }
    },

    /* ---------- 实时生成 ---------- */

    fetchRecord(studentId) {
      if (!studentId) return;
      fetch(`/api/v1/students/${studentId}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((rec) => { if (rec) this.record = rec; })
        .catch(() => {});
    },

    applyResult(rx) {
      // 展示前清洗：剔除「知识库暂无针对性资料」一类的内部字样；
      // 旧留档里只剩这类字样的风险条目（自动校验提示）没有可读内容，不再展示
      rx = sanitizePrescription(rx);
      if (Array.isArray(rx.risk_notes)) {
        rx.risk_notes = rx.risk_notes.filter((r) => (
          r.item !== '自动校验提示' && ((r.reason || '').trim() || (r.action || '').trim())
        ));
      }
      // 旧版留档没有依据链路字段：用空骨架补齐，模板不必逐处判空
      this.rx = { ...clone(EMPTY_RX), ...rx };
      // 动作详情只取这份处方自带的（后端从知识库动作条目回填），不再另拉动作库
      this.exerciseMap = {};
      (this.rx.exercise_details || []).forEach((e) => { this.exerciseMap[e.id] = e; });
      this.hasData = true;
      document.title = `${rx.name || ''}运动处方报告`;
    },

    resetState() {
      this.rx = clone(EMPTY_RX);
      this.record = null;
      this.hasData = false;
      this.genError = '';
      this.emergency = null;
      this.aiMessages = [];
      this.dayFilter = '全周';
      this.closePop();
      document.title = '运动处方报告';
    },

    async generate() {
      if (!this.selectedId || this.generating) return;
      // 报告版面挂在 v-if="!showForm" 下：先收起录入表单，结果才看得见
      this.showForm = false;
      this.resetState();
      this.generating = true;
      this.stage = '正在读取健康档案…';
      try {
        const record = await (await fetch(`/api/v1/students/${this.selectedId}`)).json();
        this.record = record;
        await this.streamGenerate(record);
      } catch (err) {
        this.genError = `生成失败：${err}`;
        this.generating = false;
        this.stage = '';
      }
    },

    /* 把一份完整档案送入处方生成流程（SSE） */
    async streamGenerate(record) {
      this.generating = true;
      this.aiOpen = true;
      this.stage = '正在生成运动处方…';
      this.abortCtrl = new AbortController();
      try {
        await streamSSE('/api/v1/prescription:generate', { record }, {
          signal: this.abortCtrl.signal,
          onEvent: (event, payload) => this.handleFrame(event, payload),
        });
      } catch (err) {
        if (err && err.name === 'AbortError') {
          this.pushAiNote('已手动停止生成。');
        } else {
          this.genError = `生成失败：${err}`;
        }
      } finally {
        this.generating = false;
        this.stage = '';
        this.abortCtrl = null;
        // 生成结果已在后端落盘：刷新 ★ 标记与「载入已有结果」按钮状态
        this.refreshCached();
      }
    },

    stopGenerate() {
      if (this.abortCtrl) this.abortCtrl.abort();
    },

    /* 跳转到第一阶段（档案理解）的报告页，携带当前学生便于直接载入 */
    openProfile() {
      if (!this.selectedId) return;
      location.href = '/profile?id=' + encodeURIComponent(this.selectedId);
    },

    /* ---------- 自行录入 ---------- */

    splitList(text) {
      return (text || '').split(/[,，、;；\n]/).map((s) => s.trim()).filter(Boolean);
    },

    buildRecord() {
      const f = this.form;
      if (!f.name || !f.age || !f.height || !f.weight) {
        this.formError = '请至少填写姓名、年龄、身高、体重';
        return null;
      }
      this.formError = '';
      const today = new Date().toISOString().slice(0, 10);
      return {
        student_id: '',
        name: f.name,
        school: '',
        physical_exam: {
          XB: f.gender, NL: f.age, SG: f.height, TZ: f.weight,
          FHL: f.fhl, SCHOOL_GRADE: f.grade, UP_TIME: today,
        },
        fitness_test: {
          NJ: f.grade, XB: f.gender, SGCJ: f.height, TZCJ: f.weight, FHLCJ: f.fhl,
          WSMCJ: f.wsm, LDTYCJ: f.ldty, ZWTQQCJ: f.zwtqq, YFZTSCJ: f.yfzts,
          WSMWFPCJ: f.wsmwfp,
          YQMCJ: f.yqm, BBMCJ: f.bbm, YTXSCJ: f.ytxs, YWQZCJ: f.ywqz,
        },
        health_profile: {
          chronic_conditions: this.splitList(f.chronic),
          medications: this.splitList(f.medications),
          injury_history: this.splitList(f.injuries).map((t) => ({ note: t })),
          recent_symptoms: this.splitList(f.symptoms),
          vitals: { resting_heart_rate: f.resting_hr, blood_pressure: f.blood_pressure },
        },
      };
    },

    async submitCustom() {
      if (this.formSaving || this.generating) return;
      const record = this.buildRecord();
      if (!record) return;

      this.formSaving = true;
      let saved;
      try {
        const res = await fetch('/api/v1/students', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(record),
        });
        if (!res.ok) {
          let detail = `HTTP ${res.status}`;
          try { detail = (await res.json()).detail || detail; } catch (e) { /* 原样提示 */ }
          this.formError = `保存失败：${detail}`;
          return;
        }
        saved = await res.json();
      } catch (err) {
        this.formError = `保存失败：${err}`;
        return;
      } finally {
        this.formSaving = false;
      }

      // 新档案已落盘：刷新下拉列表并选中，再直接走生成流程
      fetch('/api/v1/students')
        .then((r) => r.json())
        .then((list) => { this.students = list; })
        .catch(() => {});
      this.selectedId = saved.student_id;
      this.showForm = false;
      this.resetState();
      this.record = saved;
      await this.streamGenerate(saved);
    },

    /* SSE 事件处理；未知事件（如旧后端的 profile）静默忽略 */
    handleFrame(event, payload) {
      switch (event) {
        case 'status':
          this.stage = payload.message + '…';
          this.pushAiNote(payload.message + '…');
          break;
        case 'thinking':
          this.stage = '模型正在依据检索到的资料设计运动处方…';
          this.appendThink(AI_TITLE, payload.text);
          break;
        case 'token':
          this.stage = '模型正在输出处方数据…';
          this.bumpProgress('处方', payload.text.length);
          break;
        case 'retry':
          this.stage = '输出未通过校验，正在重新生成…';
          this.pushAiNote('输出未通过校验，正在重新生成…');
          break;
        case 'result':
          this.applyResult(payload);
          this.pushAiNote('运动处方生成完成。');
          break;
        case 'emergency':
          this.emergency = payload;
          this.pushAiNote('命中急症关键词，已熔断，不再调用模型。');
          break;
        case 'error':
          this.genError = payload.message;
          this.pushAiNote('生成失败：' + payload.message);
          break;
      }
    },

    /* ---------- 渲染辅助 ---------- */

    detailOf(id) { return this.exerciseMap[id] || {}; },

    categoryOf(it) {
      const d = this.detailOf(it.exercise_id);
      if (d.category) return d.category;
      // 没有编号的是常规活动（步行、慢跑等，依据来自知识库摘录而不是动作条目）
      if (!it.exercise_id) return '常规活动';
      const prefix = (it.exercise_id || '').split('-')[0];
      return PREFIX_CATEGORY[prefix] || '';
    },

    /* ---------- 依据引用 ---------- */

    /* 引用角标的悬停提示：出处，或评分结论 */
    refTitle(id) {
      const e = this.evidenceMap[id];
      if (!e) return id;
      return e.source ? `${e.source}` : id;
    },
    /* 点击依据角标：就地弹出对应条目的卡片（S = 国标评分，E = 知识库摘录） */
    openRef(id) {
      const e = this.evidenceMap[id];
      if (e) this.refPop = e;
    },
    closeRef() { this.refPop = null; },
    levelCls(level) {
      return {
        优秀: 'lv-a', 良好: 'lv-b', 及格: 'lv-c', 不及格: 'lv-d',
        正常: 'lv-a', 超重: 'lv-c', 低体重: 'lv-c', 肥胖: 'lv-d',
      }[level] || 'lv-c';
    },

    /* ---------- 动作详情弹层 ---------- */

    /* fixed 定位：贴着标签下方弹出，靠近视口底部时翻到上方 */
    placePop(evt) {
      const r = evt.currentTarget.getBoundingClientRect();
      const width = 340;
      this.pop.x = Math.max(8, Math.min(r.left, window.innerWidth - width - 12));
      this.pop.up = r.bottom > window.innerHeight - 330;
      this.pop.y = this.pop.up ? r.top - 8 : r.bottom + 8;
    },
    /* 动作详情弹层只由点击触发（悬停不弹出） */
    pillClick(evt, pill) {
      if (this.pop.pinned && this.pop.pill && this.pop.pill.key === pill.key) {
        this.closePop();
        return;
      }
      this.pop.pill = pill;
      this.placePop(evt);
      this.pop.show = true;
      this.pop.pinned = true;
    },
    closePop() {
      this.pop.show = false;
      this.pop.pinned = false;
    },

    riskCls(level) { return { 高: 'high', 中: 'mid', 低: 'low' }[level] || 'mid'; },

    parseDate(s) {
      if (!s) return null;
      const d = new Date(s);
      return isNaN(d.getTime()) ? null : d;
    },
    fmtDate(d) {
      const p = (n) => String(n).padStart(2, '0');
      return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
    },
    dateOnly(s) {
      const d = this.parseDate(s);
      return d ? this.fmtDate(d) : (s || '');
    },
  },
}).mount('#report');
