.PHONY: help demo serve init build dev test

help:
	@echo "astock-terminal"
	@echo "  make init     初始化向导（.env + 配置模板 + 数据指引）"
	@echo "  make demo     30秒演示模式（合成数据，写 ./data ./config ./reports 后起服务 :8601）"
	@echo "  make serve    启动 :8601（dist 已随仓库提供，无需 Node）"
	@echo "  make dev      前端开发模式 :5173（需另起 make dev-api）"
	@echo "  make dev-api  后端开发模式 :8602 --reload"
	@echo "  make build    重新构建前端（改了 web/frontend 代码后执行，需 Node）"
	@echo "  make test     冒烟测试（demo 数据下 API 全通）"

init:
	python -m scripts.cli.bootstrap

demo:
	python demo/generate_demo.py --install
	python -m uvicorn web.backend.app:app --host $${HOST:-127.0.0.1} --port $${PORT:-8601}

serve:
	@test -d web/frontend/dist || ($(MAKE) build)
	python -m uvicorn web.backend.app:app --host $${HOST:-127.0.0.1} --port $${PORT:-8601}

build:
	cd web/frontend && npm install && npm run build

dev:
	cd web/frontend && npm run dev

dev-api:
	python -m uvicorn web.backend.app:app --port 8602 --reload

test:
	pip install -q -r requirements-dev.txt
	python -m pytest tests/ -q
