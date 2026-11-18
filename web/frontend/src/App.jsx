import { NavLink, Outlet } from "react-router-dom";
import { Compass, Briefcase, LineChart, HeartPulse, Settings } from "lucide-react";

const NAV = [
  { to: "/", icon: Compass, label: "今日决策驾驶舱", end: true },
  { to: "/portfolio", icon: Briefcase, label: "持仓与净值中心" },
  { to: "/stock", icon: LineChart, label: "个股分析工作台" },
  { to: "/health", icon: HeartPulse, label: "系统健康" },
  { to: "/settings", icon: Settings, label: "设置" },
];

export default function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="eyebrow">Stock · Terminal</div>
          <h1>大A投研看板</h1>
          <small>只读展示层 · 数据即系统产出</small>
        </div>
        {NAV.map(({ to, icon: Icon, label, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <Icon /> {label}
          </NavLink>
        ))}
        <div className="foot">
          FastAPI + React v3<br />数据源 data/store · reports<br />⚠️ 仅供参考，不构成投资建议
        </div>
      </aside>
      <main className="main">
        <div className="content">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
