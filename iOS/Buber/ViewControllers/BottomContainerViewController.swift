//
//  BottomContainerViewController.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit
import SnapKit

final class BottomContainerViewController: UIViewController {
    enum State {
        case search
        case order
        case waiting
        case inBus
    }

    let searchViewController = SearchViewController()
    let orderViewController = OrderViewController()
    let inBusViewController = InBusViewController()

    var state: State = .search {
        didSet {
            self.updateView()
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        self.addChild(self.searchViewController)
        self.addChild(self.orderViewController)

        self.updateView()
    }

    private func updateView() {
        let animations = {
            switch self.state {
            case .search:
                self.searchViewController.clearTextField()
                self.orderViewController.view.removeFromSuperview()
                self.view.addSubview(self.searchViewController.view)
                self.searchViewController.view.snp.makeConstraints { make in
                    make.edges.equalToSuperview()
                }
            case .order:
                self.searchViewController.view.removeFromSuperview()
                self.view.addSubview(self.orderViewController.view)
                self.orderViewController.view.snp.makeConstraints { make in
                    make.edges.equalToSuperview()
                }
                self.orderViewController.setupForOrderState()
            case .waiting:
                self.orderViewController.setupForWaitingState()
            case .inBus:
                self.orderViewController.view.removeFromSuperview()
                self.view.addSubview(self.inBusViewController.view)
                self.inBusViewController.view.snp.makeConstraints { make in
                    make.edges.equalToSuperview()
                }
            }
        }

        guard self.state != .waiting else {
            animations()
            return
        }

        UIView.transition(
            with: self.view,
            duration: 0.3,
            options: .transitionCrossDissolve,
            animations: animations,
            completion: nil
        )
    }
}
